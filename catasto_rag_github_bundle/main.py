
'py main.py run_all --input_dir progettoCatasto/pdfs'

import argparse
import subprocess
from pathlib import Path

from indexer import build_index
from query_rag import interactive_chat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["extract", "index", "ask", "run_all"])
    parser.add_argument("--input_dir", type=Path, default=Path("pdfs"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--llm", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "estrazioni_local.json"
    csv_path = args.output_dir / "estrazioni_local.csv"
    index_path = args.output_dir / "index.faiss"
    docs_path = args.output_dir / "docs.json"

    if args.mode in ["extract", "run_all"]:
        subprocess.run(
            [
                "py",
                "progettoCatasto/local_rag_visure_extractor.py",
                "--input_dir",
                str(args.input_dir),
                "--output_json",
                str(json_path),
                "--output_csv",
                str(csv_path),
                "--device",
                "cpu",
            ],
            check=True,
        )

    if args.mode in ["index", "run_all"]:
        if not json_path.exists():
            raise FileNotFoundError(f"File JSON non trovato: {json_path}")

        build_index(
            input_json=json_path,
            output_index=index_path,
            output_docs=docs_path,
        )

    if args.mode in ["ask", "run_all"]:
        if not index_path.exists():
            raise FileNotFoundError(f"Indice non trovato: {index_path}")
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs file non trovato: {docs_path}")

        interactive_chat(
            index_path=index_path,
            docs_path=docs_path,
            llm_name=args.llm,
        )


if __name__ == "__main__":
    main()