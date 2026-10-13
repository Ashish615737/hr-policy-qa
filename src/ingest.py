import os
import re
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SECTION_START = re.compile(r"^\d+\.\s")


def load_pdf_text(pdf_path: Path) -> str:
    """Read a PDF and return all its text as one string."""
    reader = PdfReader(pdf_path)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def load_all_pdfs(data_dir: Path) -> dict[str, str]:
    """Return {file_name: full_text} for every PDF in the folder."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    documents: dict[str, str] = {}
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        documents[pdf_path.name] = load_pdf_text(pdf_path)
    return documents


def clean_text(text: str) -> str:
    """Remove stray spaces and re-join lines the PDF extractor wrapped."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    paragraphs: list[str] = []
    for line in lines:
        if not paragraphs or SECTION_START.match(line):
            paragraphs.append(line)
        else:
            paragraphs[-1] += " " + line

    return "\n\n".join(paragraphs)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_documents(docs: dict[str, str]) -> list[dict]:
    """Clean and chunk every document. Each chunk remembers its source file."""
    all_chunks: list[dict] = []
    for name, raw in docs.items():
        cleaned = clean_text(raw)
        for i, chunk in enumerate(chunk_text(cleaned)):
            all_chunks.append({"id": f"{name}-{i}", "source": name, "text": chunk})
    return all_chunks


if __name__ == "__main__":
    docs = load_all_pdfs(DATA_DIR)
    if not docs:
        raise SystemExit(f"No PDFs found in {DATA_DIR}")

    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"\n[{chunk['id']}] ({len(chunk['text'])} chars)")
        print(chunk["text"])