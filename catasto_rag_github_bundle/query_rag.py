import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def interactive_chat(index_path, docs_path, llm_name):

    index = faiss.read_index(str(index_path))

    with open(docs_path) as f:
        docs = json.load(f)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    llm = pipeline(
        "text-generation",
        model=llm_name,
        max_new_tokens=200
    )

    def search(query, k=3):
        q_emb = embed_model.encode([query])
        _, idx = index.search(np.array(q_emb), k)
        return [docs[i] for i in idx[0]]

    while True:
        query = input(">> ")

        results = search(query)

        context = "\n".join([str(r) for r in results])

        prompt = f"""
Usa questi dati catastali per rispondere:

{context}

Domanda: {query}
Risposta:
"""

        response = llm(prompt)[0]["generated_text"]
        print(response)