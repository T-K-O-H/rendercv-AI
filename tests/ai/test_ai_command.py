"""Tests for AI CLI commands."""

from unittest import mock

from rendercv.cli.ai_command.ai_command import (
    ai_generate,
    ai_parse,
    ai_polish,
    ai_tailor,
)


class TestAiParseCommand:
    def test_parse_creates_output_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create input file
        input_file = tmp_path / "resume.txt"
        input_file.write_text("John Doe\nSoftware Engineer")

        # Mock the AI service
        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.parse_resume.return_value = {
                "cv": {"name": "John Doe", "headline": "Software Engineer"}
            }
            mock_service.return_value = mock_instance

            ai_parse(
                input_file=input_file,
                provider="openai",
                api_key="test-key",
            )

        output_file = tmp_path / "resume_CV.yaml"
        assert output_file.exists()

    def test_parse_custom_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        input_file = tmp_path / "resume.txt"
        input_file.write_text("John Doe")
        output_file = tmp_path / "custom_output.yaml"

        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.parse_resume.return_value = {"cv": {"name": "John Doe"}}
            mock_service.return_value = mock_instance

            ai_parse(
                input_file=input_file,
                output=output_file,
                provider="openai",
                api_key="test-key",
            )

        assert output_file.exists()


class TestAiPolishCommand:
    def test_polish_creates_output_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create input YAML file
        input_file = tmp_path / "resume.yaml"
        input_file.write_text("cv:\n  name: John Doe\n")

        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.polish_resume.return_value = {
                "cv": {"name": "John Doe", "headline": "Senior Developer"}
            }
            mock_service.return_value = mock_instance

            ai_polish(
                input_file=input_file,
                provider="openai",
                api_key="test-key",
            )

        output_file = tmp_path / "resume_polished.yaml"
        assert output_file.exists()

    def test_polish_in_place(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        input_file = tmp_path / "resume.yaml"
        input_file.write_text("cv:\n  name: John Doe\n")

        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.polish_resume.return_value = {
                "cv": {"name": "John Doe Improved"}
            }
            mock_service.return_value = mock_instance

            ai_polish(
                input_file=input_file,
                in_place=True,
                provider="openai",
                api_key="test-key",
            )

        content = input_file.read_text()
        assert "John Doe Improved" in content


class TestAiTailorCommand:
    def test_tailor_creates_output_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        input_file = tmp_path / "resume.yaml"
        input_file.write_text("cv:\n  name: John Doe\n")

        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.tailor_resume.return_value = {
                "cv": {"name": "John Doe", "headline": "Python Developer"}
            }
            mock_service.return_value = mock_instance

            ai_tailor(
                input_file=input_file,
                job="Looking for a Python developer",
                provider="openai",
                api_key="test-key",
            )

        output_file = tmp_path / "resume_tailored.yaml"
        assert output_file.exists()

    def test_tailor_with_job_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        input_file = tmp_path / "resume.yaml"
        input_file.write_text("cv:\n  name: John Doe\n")

        job_file = tmp_path / "job.txt"
        job_file.write_text("We are looking for a senior engineer...")

        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.tailor_resume.return_value = {"cv": {"name": "John Doe"}}
            mock_service.return_value = mock_instance

            ai_tailor(
                input_file=input_file,
                job=str(job_file),
                provider="openai",
                api_key="test-key",
            )

        output_file = tmp_path / "resume_tailored.yaml"
        assert output_file.exists()


class TestAiGenerateCommand:
    def test_generate_creates_output_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.generate_resume.return_value = {
                "cv": {"name": "John Doe", "sections": {}}
            }
            mock_service.return_value = mock_instance

            ai_generate(
                name="John Doe",
                provider="openai",
                api_key="test-key",
            )

        output_file = tmp_path / "John_Doe_CV.yaml"
        assert output_file.exists()

    def test_generate_custom_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        output_file = tmp_path / "my_resume.yaml"

        with mock.patch(
            "rendercv.cli.ai_command.ai_command.AIResumeService"
        ) as mock_service:
            mock_instance = mock.MagicMock()
            mock_instance.generate_resume.return_value = {"cv": {"name": "John Doe"}}
            mock_service.return_value = mock_instance

            ai_generate(
                name="John Doe",
                output=output_file,
                provider="openai",
                api_key="test-key",
            )

        assert output_file.exists()
