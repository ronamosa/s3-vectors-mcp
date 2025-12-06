"""
PDF ingestion utilities for local workflows.

These helpers are intentionally light-weight: no remote downloads, no
external services beyond Bedrock + S3 Vectors, and just enough validation
for a single developer workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import PyPDF2


@dataclass
class IngestChunk:
    text: str
    start_word: int
    end_word: int


def extract_text(pdf_path: Path) -> str:
    """Pull raw text from a local PDF."""
    text_parts: List[str] = []
    with pdf_path.open("rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            text_parts.append(f"\n--- Page {idx + 1} ---\n{page_text}")
    return "".join(text_parts)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[IngestChunk]:
    """Split text into overlapping word chunks."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    words = text.split()
    step = max(chunk_size - overlap, 1)
    chunks: List[IngestChunk] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(
            IngestChunk(
                text=" ".join(chunk_words),
                start_word=start,
                end_word=min(start + chunk_size, len(words)),
            )
        )
    return chunks


def build_vector_payloads(
    pdf_path: Path,
    topic: str,
    chunks: List[IngestChunk],
    embeddings: List[List[float]],
    vault_root: Path,
) -> List[dict]:
    relative_path = (
        pdf_path.relative_to(vault_root)
        if vault_root in pdf_path.parents
        else pdf_path.name
    )
    payloads = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payloads.append(
            {
                "key": f"{topic}/{pdf_path.stem}/chunk-{idx:04d}",
                "data": {"float32": embedding},
                "metadata": {
                    "doc_name": pdf_path.name,
                    "topic": topic,
                    "chunk_id": idx,
                    "chunk_start": chunk.start_word,
                    "chunk_end": chunk.end_word,
                    "text_chunk": chunk.text[:2000],
                    "vault_path": str(relative_path),
                },
            }
        )
    return payloads

