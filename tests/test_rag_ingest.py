"""PDF 原文不会绕过人工审核直接进入 RAG。"""

from pathlib import Path

import pytest

from blood_agents.knowledge.ingest import parse_pages, stage_pdf
from blood_agents.knowledge.rag import load_approved_corpus


def test_approved_corpus_chunk_ids_are_unique_even_for_shared_source():
    chunks = load_approved_corpus()
    assert chunks
    assert len({chunk.doc_id for chunk in chunks}) == len(chunks)
    shared_source = [chunk for chunk in chunks if "2026" in chunk.source]
    assert shared_source


def test_page_ranges_use_one_based_pdf_pages():
    assert parse_pages("1,3-5,5", 6) == [0, 2, 3, 4]
    assert parse_pages(None, 2) == [0, 1]
    for invalid in ("0", "5-3", "1-7", "x", "1,,2"):
        with pytest.raises(ValueError):
            parse_pages(invalid, 6)


def test_stage_creates_unapproved_draft_and_never_overwrites(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-demo")
    output = tmp_path / "draft" / "review.md"

    def extracted(path: Path, pages: str) -> dict:
        assert path == source and pages == "2-3"
        return {
            "sha256": "a" * 64, "selected_pages": [2, 3], "extracted_chars": 8,
            "extracts": [{"page": 2, "text": "全血间隔"}, {"page": 3, "text": "180 天"}],
        }

    monkeypatch.setattr("blood_agents.knowledge.ingest.read_pdf", extracted)
    result = stage_pdf(source, "2-3", output, "https://example.gov.cn/rules.pdf")
    assert result["approved"] is False
    body = output.read_text(encoding="utf-8")
    assert "待人工审核" in body and "PDF 第 2 页" in body and "180 天" in body
    with pytest.raises(FileExistsError):
        stage_pdf(source, "2-3", output, "https://example.gov.cn/rules.pdf")


def test_stage_rejects_non_https_and_empty_extraction(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-demo")
    output = tmp_path / "review.md"
    with pytest.raises(ValueError, match="HTTPS"):
        stage_pdf(source, "1", output, "http://example.gov.cn/rules.pdf")
    monkeypatch.setattr("blood_agents.knowledge.ingest.read_pdf", lambda *_: {
        "sha256": "a" * 64, "selected_pages": [1], "extracted_chars": 0,
        "extracts": [{"page": 1, "text": ""}],
    })
    with pytest.raises(ValueError, match="扫描件"):
        stage_pdf(source, "1", output, "https://example.gov.cn/rules.pdf")
    assert not output.exists()
