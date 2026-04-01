from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Carica documento
with open("documento.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Chunking semplice
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = split_text(text)

# 3. Embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 4. Creazione embeddings
embeddings = embedder.encode(chunks)

# 5. FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 6. Query
query = "Chi è il protagonista?"
query_embedding = embedder.encode([query])

# retrieve top-k chunks
k = 3
distances, indices = index.search(np.array(query_embedding), k)
retrieved_chunks = [chunks[i] for i in indices[0]]

context = "\n".join(retrieved_chunks)

# 7. LLM
pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=200
)

# 8. Prompt
prompt = f"""
Usa il contesto per rispondere.

Contesto:
{context}

Domanda:
{query}

Risposta:
"""

response = pipe(prompt)[0]["generated_text"]

print(response)