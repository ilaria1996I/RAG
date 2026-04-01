import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_index(input_json, output_index, output_docs):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    def to_text(doc):
        return f"""
        Comune: {doc.get("comune")}
        Intestatario: {doc.get("intestatario_nome")}
        Rendita: {doc.get("rendita")}
        Foglio: {doc.get("foglio")}
        Particella: {doc.get("particella")}
        """

    texts = [to_text(d) for d in data]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, str(output_index))

    with open(output_docs, "w") as f:
        json.dump(data, f)

    print("Index creato")