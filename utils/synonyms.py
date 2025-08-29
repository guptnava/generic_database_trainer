import itertools
import streamlit as st

SYNONYM_DICT = {
    "show": ["list", "display", "give me"],
    "records": ["rows", "entries", "data"],
    "count": ["number", "total", "how many"],
    "customers": ["clients", "buyers", "users"],
    "orders": ["purchases", "transactions", "sales"],
    "employees": ["staff", "workers", "team members"],
    "products": ["items", "goods", "inventory"],
}

def generate_synonyms(question: str, max_variants: int = 5, placeholder=None):
    """
    Expand a natural language question into synonymous variants
    without using LLMs.
    """
    if placeholder is None:
        placeholder = st.empty()  # create a placeholder if not provided

    #placeholder.text(f"DEBUG: Generating synonyms for: '{question}'")
    
    words = question.split()
    variants = [[]]

    for word in words:
        key = word.lower().strip("?,.")
        #placeholder.text(f"Word: '{word}' -> key: '{key}'")
        if key in SYNONYM_DICT:
            #placeholder.text(f"  Found synonyms: {SYNONYM_DICT[key]}")
            new_variants = []
            for base in variants:
                new_variants.append(base + [word])
                for syn in SYNONYM_DICT[key]:
                    new_variants.append(base + [syn])
            variants = new_variants
        else:
            for base in variants:
                base.append(word)

    sentences = [" ".join(v) for v in variants]
    #placeholder.text(f"Generated variants: {sentences}")

    limited_sentences = list(itertools.islice(sentences, max_variants))
    #placeholder.text(f"Limited to {max_variants} variants: {limited_sentences}")

    return limited_sentences
