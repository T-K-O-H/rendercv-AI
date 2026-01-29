"""Tests for resume parser module."""

from unittest import mock

import pytest

from rendercv.ai.parser import (
    extract_resume_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
)
from rendercv.exception import RenderCVUserError


class TestExtractTextFromTxt:
    def test_extract_utf8(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello World", encoding="utf-8")

        result = extract_text_from_txt(file_path)
        assert result == "Hello World"

    def test_extract_with_special_chars(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Héllo Wörld 日本語", encoding="utf-8")

        result = extract_text_from_txt(file_path)
        assert result == "Héllo Wörld 日本語"


class TestExtractResumeText:
    def test_txt_file(self, tmp_path):
        file_path = tmp_path / "resume.txt"
        file_path.write_text("Resume content", encoding="utf-8")

        result = extract_resume_text(file_path)
        assert result == "Resume content"

    def test_md_file(self, tmp_path):
        file_path = tmp_path / "resume.md"
        file_path.write_text("# Resume\n\nContent", encoding="utf-8")

        result = extract_resume_text(file_path)
        assert result == "# Resume\n\nContent"

    def test_file_not_found(self, tmp_path):
        file_path = tmp_path / "nonexistent.txt"

        with pytest.raises(RenderCVUserError, match="File not found"):
            extract_resume_text(file_path)

    def test_unsupported_format(self, tmp_path):
        file_path = tmp_path / "resume.xyz"
        file_path.write_text("content", encoding="utf-8")

        with pytest.raises(RenderCVUserError, match="Unsupported file format"):
            extract_resume_text(file_path)

    def test_pdf_without_dependency(self, tmp_path):
        file_path = tmp_path / "resume.pdf"
        file_path.write_bytes(b"%PDF-1.4")

        # Mock import failure
        with mock.patch.dict("sys.modules", {"pypdf": None}), mock.patch(
            "builtins.__import__", side_effect=ImportError("No module named pypdf")
        ), pytest.raises(RenderCVUserError, match="pypdf package"):
            extract_text_from_pdf(file_path)

    def test_docx_without_dependency(self, tmp_path):
        file_path = tmp_path / "resume.docx"
        file_path.write_bytes(b"PK")

        with mock.patch.dict("sys.modules", {"docx": None}), mock.patch(
            "builtins.__import__", side_effect=ImportError("No module named docx")
        ), pytest.raises(RenderCVUserError, match="python-docx package"):
            extract_text_from_docx(file_path)
