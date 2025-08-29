import pandas as pd
import numpy as np
import oracledb
from scipy.spatial import KDTree
import threading
import sys
import streamlit as st
import logging
import json
import time
from sentence_transformers import SentenceTransformer

# Global variables to store the in-memory index and metadata
KD_TREE_INDEX = None
EMBEDDINGS_DATA = {}
INDEX_BUILD_LOCK = threading.Lock()


# ----------------------
# Ensure required tables exist
# ----------------------
def ensure_tables(conn):
    cur = conn.cursor()

    # NL2SQL_SCHEMA
    cur.execute("""
    BEGIN
        EXECUTE IMMEDIATE '
        CREATE TABLE NL2SQL_SCHEMA (
            id NUMBER GENERATED ALWAYS AS IDENTITY,
            schema_name VARCHAR2(128),
            table_name VARCHAR2(128),
            column_name VARCHAR2(128),
            data_type VARCHAR2(128),
            PRIMARY KEY (id)
        )';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF; -- ORA-00955 = name already used
    END;
    """)

    # NL2SQL_TRAINING
    cur.execute("""
    BEGIN
        EXECUTE IMMEDIATE '
        CREATE TABLE NL2SQL_TRAINING (
            id NUMBER GENERATED ALWAYS AS IDENTITY,
            schema_name VARCHAR2(128),
            table_name VARCHAR2(128),
            question CLOB,
            sql_template CLOB,
            PRIMARY KEY (id)
        )';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF;
    END;
    """)

    # NL2SQL_SYNONYMS
    cur.execute("""
    BEGIN
        EXECUTE IMMEDIATE '
        CREATE TABLE NL2SQL_SYNONYMS (
            id NUMBER GENERATED ALWAYS AS IDENTITY,
            training_id NUMBER,
            question_syn CLOB,
            FOREIGN KEY (training_id) REFERENCES NL2SQL_TRAINING(id)
        )';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF;
    END;
    """)

    # NL2SQL_EMBEDDINGS
    cur.execute("""
    BEGIN
        EXECUTE IMMEDIATE '
        CREATE TABLE NL2SQL_EMBEDDINGS (
            id NUMBER GENERATED ALWAYS AS IDENTITY,
            training_id NUMBER,
            question CLOB,
            embedding BLOB,
            FOREIGN KEY (training_id) REFERENCES NL2SQL_TRAINING(id)
        )';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF;
    END;
    """)

    cur.execute("""
    BEGIN
        EXECUTE IMMEDIATE '
                CREATE TABLE NL2SQL_EVALUATION (
                    id NUMBER GENERATED ALWAYS AS IDENTITY,
                    prompt CLOB,
                    expected_sql CLOB,
                    CONSTRAINT n2s_eval_pk PRIMARY KEY (id)
                )';
    EXCEPTION
    WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF;
    END;
    """)

    cur.execute("""
    BEGIN
        EXECUTE IMMEDIATE '
            CREATE TABLE NL2SQL_METRICS (
                run_id NUMBER,
                prompt_id NUMBER,
                is_hit NUMBER(1),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT n2s_metrics_fk FOREIGN KEY (prompt_id) REFERENCES NL2SQL_EVALUATION(id)
            )';
    EXCEPTION
    WHEN OTHERS THEN
            IF SQLCODE != -955 THEN RAISE; END IF;
    END;
    """)

    conn.commit()


# ----------------------
# DB Connection
# ----------------------
def connect_oracle(user, password, host, port, service):
    dsn = f"{host}:{port}/{service}"
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    ensure_tables(conn)
    return conn


# ----------------------
# Schema Extraction
# ----------------------
def insert_schema(conn, schema_owner: str):
    cur = conn.cursor()

    # Clean old rows
    cur.execute("DELETE FROM NL2SQL_SCHEMA WHERE schema_name = :owner", {"owner": schema_owner.upper()})

    # Fetch schema (excluding CLOB/BLOB)
    cur.execute("""
        SELECT owner, table_name, column_name, data_type
        FROM all_tab_columns
        WHERE owner = :owner and table_name='SALES'
          AND data_type NOT IN ('CLOB','BLOB','NCLOB','BFILE')
    """, {"owner": schema_owner.upper()})
    rows = cur.fetchall()

    # Insert into table
    cur.executemany(
        "INSERT INTO NL2SQL_SCHEMA (schema_name, table_name, column_name, data_type) VALUES (:1,:2,:3,:4)",
        rows
    )
    conn.commit()


def fetch_schema_from_db(conn):
    return pd.read_sql("SELECT * FROM NL2SQL_SCHEMA", conn)


def fetch_embeddings_from_db(conn):
    """
    Fetches embeddings data from the user's specified table structure,
    converts the binary vector and CLOB to human-readable text,
    and returns a DataFrame for display.
    """
    try:
        cur = conn.cursor()
        # Explicitly selecting columns based on user's table structure
        sql = "SELECT ID, TRAINING_ID, QUESTION, EMBEDDING FROM NL2SQL_EMBEDDINGS"
        cur.execute(sql)
        rows = cur.fetchall()
        column_names = [col[0] for col in cur.description]
        cur.close()

        # Convert the CLOB and binary vector data for display
        data_rows = []
        for row in rows:
            # Handle CLOB for 'question' column
            question_data = row[2]
            if hasattr(question_data, 'read'):
                question_text = question_data.read()
            else:
                question_text = str(question_data)

            # Handle BLOB for 'embedding' column
            embedding_data = row[3]
            if embedding_data:
                # Read the BLOB data and convert the byte array to a numpy array of floats
                embedding_bytes = embedding_data.read()
                embedding_np = np.frombuffer(embedding_bytes, dtype=np.float32)
                # Convert to a list for a more readable display
                embedding_list = embedding_np.tolist()
            else:
                embedding_list = []
            
            # Create a new row with the converted data
            new_row = [row[0], row[1], question_text, embedding_list]
            data_rows.append(new_row)
            
        # Recreate the DataFrame with the new, readable data
        df = pd.DataFrame(data_rows, columns=column_names)
        return df
    except Exception as e:
        print(f"❌ Failed to fetch embeddings data: {e}", file=sys.stderr)
        return pd.DataFrame()


# ----------------------
# Questions + Synonyms
# ----------------------
def insert_questions(conn, df):
    """
    Insert synthetic questions and SQL templates into Oracle table.
    Automatically creates the table if it does not exist.
    """
    cur = conn.cursor()
  
    cur.execute("DELETE FROM NL2SQL_SYNONYMS")
    cur.execute("DELETE FROM NL2SQL_EMBEDDINGS")
    cur.execute("DELETE FROM NL2SQL_TRAINING")
    
    conn.commit()


    # Convert DataFrame to list of tuples
    # Make sure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")

    rows = df.loc[:, ["schema_name", "table_name", "question", "sql_template"]].values.tolist()

    # Insert into Oracle
    cur.executemany(
        "INSERT INTO NL2SQL_TRAINING (schema_name, table_name, question, sql_template) VALUES (:1,:2,:3,:4)",
        rows
    )
    conn.commit()


def insert_synonyms(conn, df):
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO NL2SQL_SYNONYMS (training_id, question_syn) VALUES (:1,:2)",
        df[["training_id","question_syn"]].values.tolist()
    )
    conn.commit()

def fetch_training_data(conn):
    return pd.read_sql("SELECT id, question, sql_template FROM NL2SQL_TRAINING", conn)


def fetch_training_synonym_data(conn):
    return pd.read_sql("SELECT training_id, question_syn FROM NL2SQL_SYNONYMS", conn)


# ----------------------
# Embeddings
# ----------------------
def insert_embeddings(conn, q_df, embeddings, questOrSyn: str):
    cur = conn.cursor()
    cur.execute("DELETE FROM NL2SQL_EMBEDDINGS")  # refresh
    for i, row in q_df.iterrows():
        emb_bytes = np.asarray(embeddings[i], dtype=np.float32).tobytes()
        
        if questOrSyn == 'Quest':
            cur.execute(
                "INSERT INTO NL2SQL_EMBEDDINGS (training_id, question, embedding) VALUES (:1, :2, :3)",
                (int(row["ID"]), row["QUESTION"], emb_bytes)
            )
        else:
            cur.execute(
                "INSERT INTO NL2SQL_EMBEDDINGS (training_id, question, embedding) VALUES (:1, :2, :3)",
                (int(row["TRAINING_ID"]), row["QUESTION_SYN"], emb_bytes)
            )

    conn.commit()


def build_embedding_index(conn):
    """
    Loads all embeddings from the database into memory and builds a KD-Tree index.
    This function should be called within a lock to ensure thread-safety.
    """
    global KD_TREE_INDEX, EMBEDDINGS_DATA
    
    try:
        cur = conn.cursor()
        print("Fetching all embeddings from the database...")
        cur.execute("SELECT id, training_id, question, embedding FROM NL2SQL_EMBEDDINGS")
        rows = cur.fetchall()
        cur.close()

        print(f"Fetched {len(rows)} embeddings.")

        # Prepare data for KD-Tree and store related metadata
        embeddings_list = []
        EMBEDDINGS_DATA.clear()
        for r in rows:
            emb = np.frombuffer(r[3].read(), dtype=np.float32)
            embeddings_list.append(emb)
            # Store metadata for efficient lookup later, using the order of the fetched rows
            EMBEDDINGS_DATA[len(embeddings_list) - 1] = {"id": r[0], "training_id": r[1], "question": r[2]}
        
        # Check for empty data
        if not embeddings_list:
            print("No embeddings found to build index.")
            KD_TREE_INDEX = None
            return

        embeddings_array = np.array(embeddings_list, dtype='float32')

        print("Building KD-Tree index...")
        KD_TREE_INDEX = KDTree(embeddings_array)

        print("KD-Tree index built successfully.")

    except oracledb.Error as e:
        error, = e.args
        print(f"Error building embedding index: {error.code} - {error.message}", file=sys.stderr)
        KD_TREE_INDEX = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        KD_TREE_INDEX = None


def refresh_embedding_index(conn):
    """
    Clears the existing in-memory index and rebuilds it from the database.
    This function should be called when the underlying data has been updated.
    """
    global KD_TREE_INDEX, EMBEDDINGS_DATA
    
    with INDEX_BUILD_LOCK:
        print("Clearing existing in-memory index...")
        KD_TREE_INDEX = None
        EMBEDDINGS_DATA = {}
        print("Index cleared. Rebuilding now.")
        build_embedding_index(conn)
        print("Index refresh complete.")


def search_embeddings_kdtree(conn, query_emb, top_k=3):
    """
    Searches for the most similar embeddings using the in-memory KD-Tree index.
    The index is built automatically on the first call.
    """
    global KD_TREE_INDEX, EMBEDDINGS_DATA
    
    if KD_TREE_INDEX is None:
        print("KD-Tree index is not built. Building now...")
        build_embedding_index(conn)
        
    if KD_TREE_INDEX is None:
        # If the build failed, we can't proceed
        print("Failed to build KD-Tree index.", file=sys.stderr)
        return []

    # Perform the search on the in-memory index
    # D: Distances, I: Indices of the found embeddings
    distances, indices = KD_TREE_INDEX.query(query_emb, k=top_k)
    
    # KDTree returns a tuple of distances and indices, convert to list if only one result
    if top_k == 1:
        distances = [distances]
        indices = [indices]
    
    results = []
    # Fetch SQL templates in a single query to avoid the N+1 query problem
    db_ids_to_fetch = [EMBEDDINGS_DATA[idx]["training_id"] for idx in indices]
    
    if not db_ids_to_fetch:
        return []

    placeholders = ','.join([f':{i+1}' for i in range(len(db_ids_to_fetch))])
    
    cur = conn.cursor()
    sql_query = f"SELECT id, sql_template FROM NL2SQL_TRAINING WHERE id IN ({placeholders})"
    cur.execute(sql_query, db_ids_to_fetch)
    sql_templates = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()

    for idx, distance in zip(indices, distances):
        # We need to map the KD-Tree index back to our original database IDs
        db_data = EMBEDDINGS_DATA.get(idx)
        if not db_data:
            continue
            
        db_id = db_data["id"]
        training_id = db_data["training_id"]
        question = db_data["question"]
        sql_template = sql_templates.get(training_id)

        if sql_template:
            # Handle CLOB objects
            if hasattr(sql_template, 'read'):
                sql_template = sql_template.read()
            
            # The KDTree distance is Euclidean, not cosine similarity.
            # Convert to similarity for better user understanding.
            similarity = 1 - distance / (distance + 1)
            
            results.append({
                "question": question,
                "sql_template": sql_template,
                "similarity": float(similarity)
            })

    return results


def search_embeddings(conn, query_emb, top_k=3):
    cur = conn.cursor()
    cur.execute("SELECT id, training_id, question, embedding FROM NL2SQL_EMBEDDINGS")
    rows = cur.fetchall()

    results = []
    for r in rows:
        emb = np.frombuffer(r[3].read(), dtype=np.float32)  # convert BLOB back
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        results.append((sim, r[1], r[2]))

    results = sorted(results, key=lambda x: -x[0])[:top_k]
    out = []
    for sim, tid, q in results:
        sql = pd.read_sql(f"SELECT sql_template FROM NL2SQL_TRAINING WHERE id={tid}", conn).iloc[0,0]
        out.append({"question": q, "sql_template": sql, "similarity": float(sim)})
    return out


# Add these functions to your existing utils/oracle_utils.py file
def fetch_evaluation_prompts(conn):
    """
    Fetches a DataFrame of evaluation prompts and their expected SQL queries.
    This data is used for bulk evaluation runs. Assumes the
    NL2SQL_EVALUATION table exists.
    """
    try:
        query = "SELECT ID, PROMPT, EXPECTED_SQL FROM NL2SQL_EVALUATION"
        cur = conn.cursor()
        cur.execute(query)
        
        columns = [col[0] for col in cur.description]
        data_rows = []

        while True:
            rows = cur.fetchmany(1000)  # Fetch 1000 rows at a time
            if not rows:
                break
            for row in rows:
                # Handle CLOB objects
                prompt_text = row[1]
                if hasattr(prompt_text, 'read'):
                    prompt_text = prompt_text.read()
                
                expected_sql = row[2]
                if hasattr(expected_sql, 'read'):
                    expected_sql = expected_sql.read()
                
                data_rows.append([row[0], prompt_text, expected_sql])

        cur.close()
        df = pd.DataFrame(data_rows, columns=columns)
        return df
    except oracledb.Error as e:
        error, = e.args
        print(f"Error fetching evaluation prompts: {error.code} - {error.message}", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return pd.DataFrame()


def insert_evaluation_metric(conn, run_id, prompt_id, is_hit):
    """
    Inserts a single evaluation metric record into the NL2SQL_METRICS table.
    Records whether a specific prompt was a 'hit' (1) or a 'miss' (0).
    """
    try:
        cur = conn.cursor()
        sql = "INSERT INTO NL2SQL_METRICS (RUN_ID, PROMPT_ID, IS_HIT) VALUES (:1, :2, :3)"
        cur.execute(sql, [run_id, prompt_id, 1 if is_hit else 0])
        conn.commit()
        cur.close()
        return True
    except oracledb.Error as e:
        error, = e.args
        print(f"Error inserting evaluation metric: {error.code} - {error.message}", file=sys.stderr)
        conn.rollback()
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        conn.rollback()
        return False


def fetch_evaluation_metrics(conn):
    """
    Fetches all records from the NL2SQL_METRICS table and returns them
    as a pandas DataFrame for dashboard display.
    """
    try:
        query = "SELECT RUN_ID, PROMPT_ID, IS_HIT, TIMESTAMP FROM NL2SQL_METRICS"
        cur = conn.cursor()
        cur.execute(query)
        
        columns = [col[0] for col in cur.description]
        data_rows = []

        while True:
            rows = cur.fetchmany(1000) # Fetch 1000 rows at a time
            if not rows:
                break
            data_rows.extend(rows)

        cur.close()
        df = pd.DataFrame(data_rows, columns=columns)
        return df
    except oracledb.Error as e:
        error, = e.args
        print(f"Error fetching evaluation metrics: {error.code} - {error.message}", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return pd.DataFrame()
