# utils/synthetic_questions.py

import random

QUESTION_TEMPLATES = {
    "select": [
        "Show all records from {table}",
        "List everything in {table}",
        "Retrieve all rows in {table}",
    ],
    "count": [
        "How many records are in {table}?",
        "Give me the total number of rows in {table}",
    ],
    "filter": [
        "Show all {table} where {column} = '{value}'",
        "List {table} entries with {column} equal to '{value}'",
    ],
    "groupby": [
        "Count {table} grouped by {column}",
        "Show number of rows in {table} for each {column}",
    ],
    "join": [
        "Show {t1} joined with {t2} on {t1}.{col1} = {t2}.{col2}",
        "List all records by joining {t1} and {t2} where {t1}.{col1} = {t2}.{col2}",
    ],
}

def generate_questions(tables: list) -> list[dict]:
    """
    Generate synthetic NL->SQL questions for given tables.

    Args:
        tables: List of table dicts, each with keys: schema_name, table_name, columns

    Returns:
        List of dicts with keys: schema_name, table_name, question, sql_template
    """
    all_questions = []

    for table in tables:
        schema = table["schema_name"]
        tname = table["table_name"]
        cols = [col["name"] for col in table["columns"]]

        if not cols:
            continue  # skip tables with no usable columns

        # Basic SQL templates
        for col in cols:
            all_questions.append({
                "schema_name": schema,
                "table_name": tname,
                "question": f"Show me all {col} from {tname}",
                "sql_template": f"SELECT {col} FROM {schema}.{tname}"
            })
        
        # Example: SELECT with WHERE
        if len(cols) >= 2:
            all_questions.append({
                "schema_name": schema,
                "table_name": tname,
                "question": f"Show me {cols[0]} where {cols[1]} = ?",
                "sql_template": f"SELECT {cols[0]} FROM {schema}.{tname} WHERE {cols[1]} = :1"
            })
        
        # Example: GROUP BY
        if len(cols) >= 2:
            all_questions.append({
                "schema_name": schema,
                "table_name": tname,
                "question": f"Count {cols[0]} grouped by {cols[1]}",
                "sql_template": f"SELECT {cols[1]}, COUNT({cols[0]}) FROM {schema}.{tname} GROUP BY {cols[1]}"
            })

    return all_questions

