import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# carica dati
with open("estrazioni_local.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# trasformazione in testo
def to_text(doc):
    return f"""
    Immobile nel comune di {doc.get("comune")}
    intestato a {doc.get("intestatario_nome")}
    rendita {doc.get("rendita")}
    foglio {doc.get("foglio")}
    particella {doc.get("particella")}
    """

texts = [to_text(d) for d in data]

# embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# salva
faiss.write_index(index, "index.faiss")

# salva mapping
with open("docs.json", "w") as f:
    json.dump(data, f)