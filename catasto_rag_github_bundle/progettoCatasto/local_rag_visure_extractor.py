
"""
Run:
py progettoCatasto/local_rag_visure_extractor.py --input_dir progettoCatasto/pdfs
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import pipeline


@dataclass
class Chunk:
    pdf_path: str
    page_num: int
    chunk_id: int
    text: str


FIELDS = [
    "file_name",
    "comune",
    "codice_comune",
    "provincia",
    "foglio",
    "particella",
    "subalterno",
    "categoria",
    "classe",
    "consistenza",
    "superficie_catastale",
    "superficie_escluse_aree_scoperte",
    "rendita",
    "indirizzo",
    "piano",
    "interno",
    "intestatario_nome",
    "intestatario_data_nascita",
    "intestatario_luogo_nascita",
    "codice_fiscale",
    "diritti_e_oneri_reali",
    "dati_derivanti_da",
    "tipo_atto",
    "data_atto",
    "data_registrazione",
    "volume",
    "numero",
    "sede",
    "protocollo",
    "mappali_terreni_correlati",
    "unita_immobiliari_n",
    "tributi_erariali",
    "visura_numero",
    "data_visura",
    "ora_visura",
    "generated_by",
    "operatore",
    "note",
    "confidence_note",
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            pages.append((page_idx, clean_text(txt)))
    return pages


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks


def build_chunks(pdf_path: Path, pages: List[Tuple[int, str]], chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 0

    for page_num, text in pages:
        if not text:
            continue

        raw_chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        for piece in raw_chunks:
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(str(pdf_path), page_num, chunk_id, piece))
                chunk_id += 1

    return chunks


def regex_prefill(full_text: str) -> Dict[str, Optional[str]]:
    patterns = {
        "visura_numero": r"Visura n\.\s*([A-Z0-9]+)",
        "data_visura": r"Data:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})",
        "ora_visura": r"Ora:\s*([0-9\.:]+)",
        "comune": r"Comune di\s+([A-ZÀ-Üa-zà-ü' ]+)",
        "codice_comune": r"Comune di .*?\( ?Codice:\s*([A-Z0-9]+)\)",
        "provincia": r"Provincia di\s+([A-ZÀ-Üa-zà-ü' ]+)",
        "foglio": r"Foglio:\s*([A-Z0-9]+)",
        "particella": r"Particella:\s*([A-Z0-9]+)",
        "subalterno": r"(?:Sub\.?|Subalterno)\s*:?\s*([A-Z0-9]+)",
        "categoria": r"\b([A-E]/\d+)\b",
        "classe": r"Classe\s*:?\s*([0-9A-Z]+)",
        "consistenza": r"([0-9]+(?:,[0-9]+)?\s*vani)",
        "superficie_catastale": r"Totale:\s*([0-9]+(?:[,\.][0-9]+)?\s*m²?)",
        "superficie_escluse_aree_scoperte": r"escluse aree scoperte\*\*:\s*([0-9]+(?:[,\.][0-9]+)?\s*m²?)",
        "rendita": r"(Euro\s*[0-9\.\,]+|€\s*[0-9\.\,]+)",
        "tributi_erariali": r"Tributi erariali:\s*(Euro\s*[0-9\.\,]+|€\s*[0-9\.\,]+)",
        "codice_fiscale": r"\b([A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z])\b",
        "protocollo": r"Protocollo\s*:?\s*([0-9]+)",
        "volume": r"Volume:\s*([0-9]+)",
        "numero": r"Numero:\s*([0-9]+)",
        "sede": r"Sede:\s*([A-ZÀ-Üa-zà-ü' ]+)",
        "operatore": r"Operatore:\s*([A-Z0-9_\- ]+)",
        "generated_by": r"Documento generato da:\s*(.+)",
        "note": r"Nota:\s*(.+)",
    }

    out: Dict[str, Optional[str]] = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, full_text, flags=re.IGNORECASE)
        out[key] = m.group(1).strip() if m else None
    return out


class LocalRAGExtractor:
    def __init__(
        self,
        embed_model_name: str,
        llm_name: str,
        device: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.embedder = SentenceTransformer(embed_model_name)

        model_kwargs = {}
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        self.generator = pipeline(
            "text-generation",
            model=llm_name,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )

    def build_index(self, chunks: List[Chunk]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, embeddings

    def retrieve(self, query: str, index: faiss.IndexFlatL2, chunks: List[Chunk], k: int = 5) -> List[Chunk]:
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = index.search(query_embedding, min(k, len(chunks)))
        return [chunks[i] for i in indices[0] if i >= 0]

    def build_context(self, index: faiss.IndexFlatL2, chunks: List[Chunk]) -> str:
        queries = [
            "dati della richiesta comune provincia foglio particella subalterno",
            "unità immobiliare categoria classe consistenza superficie catastale rendita indirizzo piano interno",
            "intestato intestatario codice fiscale diritti e oneri reali",
            "dati derivanti da tipo atto data atto data registrazione volume numero sede protocollo",
            "tributi erariali visura numero data ora operatore documento generato da nota",
        ]

        selected = []
        seen = set()

        for query in queries:
            for chunk in self.retrieve(query, index, chunks, k=5):
                key = (chunk.page_num, chunk.chunk_id)
                if key not in seen:
                    seen.add(key)
                    selected.append(chunk)

        parts = []
        for chunk in selected[:15]:
            parts.append(f"[PAGINA {chunk.page_num} | CHUNK {chunk.chunk_id}]\n{chunk.text}")

        return "\n\n---\n\n".join(parts)

    def extract_json(self, file_name: str, prefill: Dict[str, Optional[str]], context: str) -> Dict[str, Optional[str]]:
        schema_text = ", ".join(FIELDS)

        prompt = f"""
Sei un estrattore documentale specializzato in visure catastali italiane.

Devi estrarre i dati dal contesto e restituire SOLO JSON valido.
Regole:
- Nessun testo fuori dal JSON.
- Usa esattamente queste chiavi: {schema_text}
- Se un campo non è presente, usa null.
- Non inventare valori.
- confidence_note deve contenere una breve nota su campi mancanti o dubbi.

FILE: {file_name}

PREFILL:
{json.dumps(prefill, ensure_ascii=False, indent=2)}

CONTESTO:
{context}

JSON:
"""

        response = self.generator(prompt, return_full_text=False)[0]["generated_text"].strip()

        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Il modello non ha restituito JSON valido.\nOutput:\n{response}")

        data = json.loads(match.group(0))

        normalized = {field: data.get(field) for field in FIELDS}
        normalized["file_name"] = file_name
        return normalized


def process_pdf(
    extractor: LocalRAGExtractor,
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
) -> Dict[str, Optional[str]]:
    pages = extract_pdf_text(pdf_path)
    full_text = "\n\n".join(text for _, text in pages).strip()

    if not full_text:
        return {field: None for field in FIELDS} | {
            "file_name": pdf_path.name,
            "confidence_note": "PDF senza testo estraibile. Probabile scansione: serve OCR."
        }

    chunks = build_chunks(pdf_path, pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {field: None for field in FIELDS} | {
            "file_name": pdf_path.name,
            "confidence_note": "Nessun chunk costruito dal PDF."
        }

    index, _ = extractor.build_index(chunks)
    prefill = regex_prefill(full_text)
    context = extractor.build_context(index, chunks)
    result = extractor.extract_json(pdf_path.name, prefill, context)

    for key, value in prefill.items():
        if key in result and not result.get(key) and value:
            result[key] = value

    if result.get("file_name") is None:
        result["file_name"] = pdf_path.name

    return result


def save_results(results: List[Dict[str, Optional[str]]], output_json: Path, output_csv: Path) -> None:
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")


def find_pdfs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG locale per estrarre dati da visure PDF.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Cartella contenente i PDF.")
    parser.add_argument("--output_json", type=Path, default=Path("estrazioni_local.json"))
    parser.add_argument("--output_csv", type=Path, default=Path("estrazioni_local.csv"))
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Modello embedding."
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="LLM locale via transformers pipeline."
    )
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Cartella non trovata: {args.input_dir}")

    pdfs = find_pdfs(args.input_dir)
    if not pdfs:
        raise FileNotFoundError(f"Nessun PDF trovato in: {args.input_dir}")

    extractor = LocalRAGExtractor(
        embed_model_name=args.embed_model,
        llm_name=args.llm,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    results: List[Dict[str, Optional[str]]] = []
    for pdf_path in tqdm(pdfs, desc="Estrazione"):
        try:
            results.append(process_pdf(extractor, pdf_path, args.chunk_size, args.overlap))
        except Exception as e:
            results.append(
                {field: None for field in FIELDS} | {
                    "file_name": pdf_path.name,
                    "confidence_note": f"Errore: {type(e).__name__}: {e}"
                }
            )

    save_results(results, args.output_json, args.output_csv)

    print(f"Processati {len(results)} PDF")
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
