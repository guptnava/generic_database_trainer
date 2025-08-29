import logging
import streamlit as st
import pandas as pd
import numpy as np
import oracledb
import json
import time
from sentence_transformers import SentenceTransformer
from utils.oracle_utils import (
    connect_oracle, insert_schema, insert_questions,
    insert_synonyms, insert_embeddings, search_embeddings_kdtree,search_embeddings,
    fetch_schema_from_db, fetch_training_data, fetch_training_synonym_data,
    fetch_embeddings_from_db,
    # New functions for evaluation (assumed to be in your utils file)
    fetch_evaluation_prompts, insert_evaluation_metric, fetch_evaluation_metrics,
    # New function to refresh the index
    refresh_embedding_index
)
from utils.synthetic_questions import generate_questions
from utils.synonyms import generate_synonyms

# Load model once
@st.cache_resource
def load_model():
    # Use a local path or a valid model name
    return SentenceTransformer("../local_all-MiniLM-L6-v2")

model = load_model()

# Setup logging
log_file = "synonym_generation.log"
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w"  # overwrite each run; use "a" to append
)


# Sidebar for DB config
st.sidebar.header("Oracle Connection")
host = st.sidebar.text_input("Host", "localhost")
port = st.sidebar.text_input("Port", "1521")
service = st.sidebar.text_input("Service", "riskintegov2")
user = st.sidebar.text_input("User", "riskintegov2")
password = st.sidebar.text_input("Password", "riskintegov2", type="password")

if st.sidebar.button("Connect"):
    try:
        conn = connect_oracle(user, password, host, port, service)
        st.session_state["conn"] = conn
        st.sidebar.success("✅ Connected to Oracle")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {e}")

st.title("🧠 NLP → MDS Training & Evaluation Centre")

tab1, tab2, tab3, tab4 = st.tabs(["1. Scan Schema", "2. Generate Questions", "3. Build Embeddings", "4. Evaluation"])

# -------------------------
# TAB 1: Extract Schema
# -------------------------
with tab1:
    st.header("Extract Oracle Schema")

    schema_owner = st.text_input("Enter schema owner (Oracle user):")

    if st.button("Extract & Store Schema"):
        with st.spinner(f"Extracting schema for {schema_owner} from Oracle..."):
            conn = st.session_state.get("conn")
            if not conn:
                st.error("Not connected to Oracle")
            elif not schema_owner.strip():
                st.warning("Please enter a schema owner before extracting.")
            else:
                try:
                    # Call helper with schema owner
                    insert_schema(conn, schema_owner)

                    st.success(f"✅ Schema for `{schema_owner}` stored in Oracle table NL2SQL_SCHEMA")

                    # Display stored schema
                    df = fetch_schema_from_db(conn)
                    st.dataframe(df)

                except Exception as e:
                    st.error(f"❌ Failed to extract schema: {e}")

    # Add a button to view the stored schema without re-extracting
    if st.button("View Stored Schema"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            try:
                st.subheader("Current Schema Data")
                df = fetch_schema_from_db(conn)
                if not df.empty:
                    st.dataframe(df)
                else:
                    st.info("No schema data found. Please extract the schema first.")
            except Exception as e:
                st.error(f"❌ Failed to fetch schema data: {e}")


# -------------------------
# TAB 2: Generate Questions
# -------------------------
with tab2:
    st.header("Generate Synthetic Questions + Synonyms")

    if st.button("Generate & Store Questions"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            try:
                # Fetch schema from Oracle
                schema_df = fetch_schema_from_db(conn)
                st.success(f"Schema records fetched: {len(schema_df)}")

                # Group columns by table
                tables = []
                for (schema_name, table_name), group in schema_df.groupby(["SCHEMA_NAME", "TABLE_NAME"]):
                    table_info = {
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "columns": [
                            {"name": row["COLUMN_NAME"], "type": row["DATA_TYPE"]}
                            for _, row in group.iterrows()
                            if row["DATA_TYPE"].upper() not in ("CLOB", "BLOB")  # ignore LOBs
                        ]
                    }
                    if table_info["columns"]:  # only include tables with columns
                        tables.append(table_info)

                if not tables:
                    st.warning("⚠️ No valid tables found in schema.")
                else:
                    # Generate synthetic questions
                    with st.spinner("Generating questions..."):
                        q_list = generate_questions(tables)
                    
                    if not q_list:
                        st.warning("⚠️ No questions were generated. Skipping synonyms.")
                    else:
                        # Convert to DataFrame for Oracle insert
                        q_df = pd.DataFrame(q_list)
                        insert_questions(conn, q_df)
                        st.success(f"✅ {len(q_df)} questions stored in Oracle")
                        st.dataframe(q_df.head(20))

                        # Fetch back question IDs to maintain relationship
                        cur = conn.cursor()
                        cur.execute("SELECT id, question FROM NL2SQL_TRAINING")
                        questions_in_db = cur.fetchall()  # list of tuples (training_id, question)
                        logging.info(f"Fetched {questions_in_db} questions from Oracle.")
                        cur.close()


                        # Generate synonyms
                        with st.spinner("Generating synonyms..."):
                            syn_list = []
                            for training_id, question in questions_in_db:
                                 #Ensure question is a string
                                if hasattr(question, "read"):
                                        question = question.read()  # convert LOB to string

                                 # Debug: show the question being processed
                                logging.info(f"Fetched {question} questions from Oracle.")
                                

                                synonyms = generate_synonyms(question)  
                                logging.info(f"Fetched {synonyms} synonyms from function.")
                                                            
                                if synonyms:
                                    for syn in synonyms:
                                        syn_list.append({
                                            "training_id": training_id,
                                            "question_syn": syn
                                        })

                            if syn_list:
                                syn_df = pd.DataFrame(syn_list)
                                insert_synonyms(conn, syn_df)
                                st.success(f"✅ {len(syn_df)} synonyms stored in Oracle")
                            else:
                                st.warning("⚠️ No synonyms generated")

            except Exception as e:
                st.error(f"❌ Failed to generate/store questions: {str(e)}")

    # Add a button to view the stored questions and synonyms
    if st.button("View Questions and Synonyms"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            try:
                st.subheader("Current Training Questions")
                q_df = fetch_training_data(conn)
                if not q_df.empty:
                    st.dataframe(q_df)
                else:
                    st.info("No training questions found. Please generate them first.")

                st.subheader("Current Training Synonyms")
                syn_df = fetch_training_synonym_data(conn)
                if not syn_df.empty:
                    st.dataframe(syn_df)
                else:
                    st.info("No synonyms found. Please generate them first.")

            except Exception as e:
                st.error(f"❌ Failed to fetch questions and synonyms: {e}")



# -------------------------
# TAB 3: Build Embeddings
# -------------------------
with tab3:
    st.header("Generate & Store Embeddings")
    if st.button("Build Embeddings"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            q_df = fetch_training_data(conn)
            
            # Convert CLOBs to strings
            q_df['question'] = q_df['QUESTION'].apply(lambda x: x.read() if hasattr(x, 'read') else str(x))
            
            texts = q_df["question"].tolist()
            embeddings = model.encode(texts, convert_to_numpy=True)
            insert_embeddings(conn, q_df, embeddings, 'Quest')
            
            st.success(f"✅ Stored {len(texts)} embeddings into NL2SQL_EMBEDDINGS")

            q_df_syn = fetch_training_synonym_data(conn)
            
            # Convert CLOBs to strings
            q_df_syn['question_syn'] = q_df_syn['QUESTION_SYN'].apply(lambda x: x.read() if hasattr(x, 'read') else str(x))
            
            texts = q_df_syn["question_syn"].tolist()
            embeddings = model.encode(texts, convert_to_numpy=True)
            insert_embeddings(conn, q_df_syn, embeddings , 'Syn')
            
            st.success(f"✅ Stored {len(texts)} synonym embeddings into NL2SQL_EMBEDDINGS")

    # Add a button to view stored embeddings (excluding the large binary vector)
    if st.button("View Stored Embeddings"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            try:
                st.subheader("Current Embeddings Data")
                df = fetch_embeddings_from_db(conn)
                if not df.empty:
                    st.dataframe(df)
                else:
                    st.info("No embeddings found. Please build them first.")
            except Exception as e:
                st.error(f"❌ Failed to fetch embeddings data: {e}")

    if st.button("Rebuild Index"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            with st.spinner("Rebuilding in-memory index..."):
                refresh_embedding_index(conn)
            st.success("✅ Index rebuilt successfully!")


# -------------------------
# TAB 4: Evaluation
# -------------------------
with tab4:
    st.header("Evaluate User Prompt → SQL Retrieval")
    user_prompt = st.text_area("Enter natural language query")
    k = st.slider("Top-K", 1, 10, 3)

    if st.button("Evaluate Prompt"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            q_emb = model.encode([user_prompt], convert_to_numpy=True)[0]
            results = search_embeddings_kdtree(conn, q_emb, top_k=k)
            st.subheader("Retrieved SQL Candidates")
            for r in results:
                st.markdown(f"**Q:** {r['question']} (Similarity: **{r['similarity']:.2f}**)")
                st.code(r["sql_template"], language="sql")

    st.markdown("---")
    st.header("Bulk Evaluation")

    if st.button("Run Bulk Evaluation"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            try:
                with st.spinner("Running bulk evaluation..."):
                    # Fetch evaluation prompts from a new Oracle table
                    eval_df = fetch_evaluation_prompts(conn)
                    total_runs = len(eval_df)
                    if total_runs == 0:
                        st.warning("No evaluation prompts found. Please populate the `NL2SQL_EVALUATION` table.")
                    else:
                        hits = 0
                        run_id = int(time.time()) # Use timestamp as a unique run ID
                        
                        # Loop through each evaluation prompt
                        for index, row in eval_df.iterrows():
                            prompt_id = row['ID']
                            prompt_text = row['PROMPT']
                            expected_sql = row['EXPECTED_SQL']

                            # Get top-k SQL candidates from the vector search
                            q_emb = model.encode([prompt_text], convert_to_numpy=True)[0]
                            results = search_embeddings_kdtree(conn, q_emb, top_k=5) # Using top_k=5 for evaluation

                            # Check if the expected SQL is in the top-k results
                            is_hit = any(res['sql_template'].strip().replace("'", '"').lower() == expected_sql.strip().replace("'", '"').lower() for res in results)
                            if is_hit:
                                hits += 1

                            # Log the result to the metrics table
                            insert_evaluation_metric(conn, run_id, prompt_id, is_hit)

                        hit_rate = (hits / total_runs) * 100
                        st.success(f"✅ Bulk evaluation complete. Total runs: {total_runs}, Hits: {hits}, Hit Rate: {hit_rate:.2f}%")
            except Exception as e:
                st.error(f"❌ Failed to run bulk evaluation: {e}")

    if st.button("View Evaluation Metrics"):
        conn = st.session_state.get("conn")
        if not conn:
            st.error("Not connected to Oracle")
        else:
            try:
                st.subheader("Evaluation Metrics Dashboard")
                metrics_df = fetch_evaluation_metrics(conn)
                if not metrics_df.empty:
                    # Calculate summary stats
                    total_runs = len(metrics_df)
                    total_hits = metrics_df['IS_HIT'].sum()
                    hit_rate_pct = (total_hits / total_runs) * 100 if total_runs > 0 else 0

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Total Evaluation Runs", value=total_runs)
                    with col2:
                        st.metric(label="Overall Hit Rate", value=f"{hit_rate_pct:.2f}%")

                    st.dataframe(metrics_df.sort_values(by='RUN_ID', ascending=False))
                else:
                    st.info("No evaluation metrics found. Please run a bulk evaluation first.")
            except Exception as e:
                st.error(f"❌ Failed to fetch evaluation metrics: {e}")
