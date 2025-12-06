from pathlib import Path

import pytest

from s3_vectors_mcp import ingest


def test_chunk_text_respects_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = ingest.chunk_text(text, chunk_size=20, overlap=5)
    assert chunks, "Should create chunks"
    assert chunks[0].end_word - chunks[0].start_word == 20
    # step should be 15
    assert chunks[1].start_word - chunks[0].start_word == 15


def test_chunk_text_validates_inputs():
    with pytest.raises(ValueError):
        ingest.chunk_text("hello world", chunk_size=10, overlap=10)


def test_build_vector_payloads_truncates_text(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("dummy")
    chunks = [ingest.IngestChunk(text="abc" * 2000, start_word=0, end_word=10)]
    vectors = ingest.build_vector_payloads(
        pdf,
        topic="demo",
        chunks=chunks,
        embeddings=[[0.1, 0.2]],
        vault_root=tmp_path,
    )
    assert "text_chunk" in vectors[0]["metadata"]
    assert len(vectors[0]["metadata"]["text_chunk"]) <= 2000

