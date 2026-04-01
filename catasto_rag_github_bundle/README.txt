# 🏠 Catasto Intelligente — RAG System per Documenti Catastali

## 📌 Overview

Questo progetto implementa una pipeline completa per:

1. Estrarre dati strutturati da PDF catastali
2. Costruire un indice semantico (vector database)
3. Interrogare i dati tramite un sistema RAG (Retrieval-Augmented Generation)

L’obiettivo è dimostrare l’uso corretto delle RAG distinguendo tra:

* **document extraction**
* **semantic retrieval**
* **reasoning su più documenti**

---

## 🧠 Architettura

```text
PDF → Text Extraction → JSON → Embeddings → FAISS → Retrieval → LLM → Answer
```

---

## ⚙️ Tecnologie utilizzate

* Python
* pdfplumber (text extraction)
* sentence-transformers (embeddings)
* FAISS (vector search)
* transformers (LLM locale)
* pandas (export dati)

---

## 📂 Struttura del progetto

```text
project/
│
├── progettoCatasto/
│   ├── pdfs/                     # PDF di input
│   ├── local_rag_visure_extractor.py
│
├── build_index.py               # costruzione embeddings + FAISS
├── query_rag.py                 # sistema RAG interattivo
├── main.py                      # orchestratore (pipeline completa)
│
├── outputs/
│   ├── estrazioni_local.json
│   ├── estrazioni_local.csv
│   ├── index.faiss
│   └── docs.json
```

---

## 🚀 Setup

### 1. Creazione ambiente (consigliato)

```bash
py -m venv venv
venv\Scripts\activate
```

### 2. Installazione dipendenze

```bash
pip install pdfplumber sentence-transformers faiss-cpu transformers torch pandas tqdm accelerate
```

---

## ▶️ Esecuzione

### 🔹 Pipeline completa

```bash
py main.py run_all --input_dir progettoCatasto/pdfs
```

---

### 🔹 Solo estrazione

```bash
py main.py extract --input_dir progettoCatasto/pdfs
```

---

### 🔹 Solo indicizzazione

```bash
py main.py index
```

---

### 🔹 Query RAG

```bash
py main.py ask
```

---

## 🧪 Esempi di query

```text
Chi possiede immobili a Comune_1?
Qual è la rendita più alta?
Quanti immobili ha Mario Rossi?
Trova immobili simili
```

---

## 🧠 Scelte progettuali

### ❗ Separazione Extraction vs RAG

L’estrazione dei dati dai PDF non utilizza RAG, ma una pipeline dedicata.

Motivazione:

* ogni PDF contiene già tutte le informazioni
* RAG è utile quando i dati sono distribuiti tra documenti

👉 Il sistema utilizza RAG **solo nella fase di interrogazione**

---

### 🔍 Retrieval semantico

* Embedding: sentence-transformers
* Similarità: FAISS
* Top-K retrieval per costruzione del contesto

---

### 🤖 Generazione

Un modello LLM locale viene utilizzato per:

* interpretare il contesto recuperato
* generare risposte coerenti

---

## 📊 Risultati

Esempio estrazione
{
  "file_name": "visura_realistica_1.pdf",
  "comune": "Comune_1",
  "foglio": "123",
  "particella": "456",
  "rendita": "€1200",
  "intestatario_nome": "Mario Rossi"
}
🔹 Esempio query RAG
Query:
Chi possiede immobili nel Comune_1?

Risposta:
Mario Rossi possiede immobili nel Comune_1 con una rendita di €1200.
🔹 Esempio multi-documento
Query:
Quanti immobili possiede Mario Rossi?

Risposta:
Mario Rossi possiede 3 immobili distribuiti in più comuni.
```

---

## ⚠️ Limitazioni

* Prestazioni limitate su CPU
* Accuratezza dipendente dalla qualità del parsing PDF
* Modelli piccoli → capacità di reasoning limitata

---

## 🚀 Possibili miglioramenti

* Hybrid search (keyword + semantic)
* Filtri strutturati (tipo SQL)
* UI (Streamlit / web app)
* Evaluation automatica (precision/recall)
* Reranking dei risultati

---


## 👤 Autore

Ilaria Figliuzzi

---
