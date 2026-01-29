"""Tests for AI service module."""

from unittest import mock

import pytest

from rendercv.ai.service import (
    AIResumeService,
    load_resume_yaml,
    read_job_description,
    save_resume_yaml,
)
from rendercv.exception import RenderCVUserError


class TestAIResumeService:
    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        with mock.patch("rendercv.ai.service.get_ai_client") as mock_get_client:
            mock_client = mock.MagicMock()
            mock_get_client.return_value = mock_client
            yield mock_client

    def test_parse_resume(self, mock_ai_client, tmp_path):
        # Create a test resume file
        resume_file = tmp_path / "resume.txt"
        resume_file.write_text("John Doe\nSoftware Engineer\ntest@email.com")

        # Mock AI response
        mock_ai_client.generate_json.return_value = {
            "cv": {
                "name": "John Doe",
                "headline": "Software Engineer",
                "email": "test@email.com",
            }
        }

        service = AIResumeService(provider="openai", api_key="test-key")
        result = service.parse_resume(resume_file)

        assert "cv" in result
        assert result["cv"]["name"] == "John Doe"

    def test_parse_resume_empty_file(self, mock_ai_client, tmp_path):  # noqa: ARG002
        resume_file = tmp_path / "resume.txt"
        resume_file.write_text("")

        service = AIResumeService(provider="openai", api_key="test-key")

        with pytest.raises(RenderCVUserError, match="Could not extract any text"):
            service.parse_resume(resume_file)

    def test_parse_resume_invalid_response(self, mock_ai_client, tmp_path):
        resume_file = tmp_path / "resume.txt"
        resume_file.write_text("Some content")

        # Mock invalid AI response (missing 'cv' key)
        mock_ai_client.generate_json.return_value = {"invalid": "response"}

        service = AIResumeService(provider="openai", api_key="test-key")

        with pytest.raises(RenderCVUserError, match="failed to generate valid"):
            service.parse_resume(resume_file)

    def test_polish_resume(self, mock_ai_client):
        resume_data = {
            "cv": {
                "name": "John Doe",
                "sections": {"experience": [{"company": "Test", "position": "Dev"}]},
            }
        }

        mock_ai_client.generate_json.return_value = {
            "cv": {
                "name": "John Doe",
                "sections": {
                    "experience": [
                        {
                            "company": "Test",
                            "position": "Software Developer",
                            "highlights": ["Improved performance by 50%"],
                        }
                    ]
                },
            }
        }

        service = AIResumeService(provider="openai", api_key="test-key")
        result = service.polish_resume(resume_data)

        assert "cv" in result
        assert result["cv"]["sections"]["experience"][0]["highlights"]

    def test_polish_resume_with_focus_areas(self, mock_ai_client):
        resume_data = {"cv": {"name": "John Doe"}}

        mock_ai_client.generate_json.return_value = {"cv": {"name": "John Doe"}}

        service = AIResumeService(provider="openai", api_key="test-key")
        service.polish_resume(resume_data, focus_areas=["action verbs", "metrics"])

        # Verify the prompt included focus areas
        call_args = mock_ai_client.generate_json.call_args
        assert "action verbs" in call_args.kwargs.get("prompt", call_args[0][0])

    def test_tailor_resume(self, mock_ai_client):
        resume_data = {"cv": {"name": "John Doe"}}
        job_description = "Looking for a Python developer..."

        mock_ai_client.generate_json.return_value = {
            "cv": {"name": "John Doe", "headline": "Python Developer"}
        }

        service = AIResumeService(provider="openai", api_key="test-key")
        result = service.tailor_resume(resume_data, job_description)

        assert "cv" in result
        assert result["cv"]["headline"] == "Python Developer"

    def test_tailor_resume_empty_job_description(
        self, mock_ai_client  # noqa: ARG002
    ):
        resume_data = {"cv": {"name": "John Doe"}}

        service = AIResumeService(provider="openai", api_key="test-key")

        with pytest.raises(RenderCVUserError, match="Job description cannot be empty"):
            service.tailor_resume(resume_data, "   ")

    def test_generate_resume(self, mock_ai_client):
        user_info = "Name: John Doe\nExperience: 5 years in software development"

        mock_ai_client.generate_json.return_value = {
            "cv": {
                "name": "John Doe",
                "headline": "Senior Software Developer",
                "sections": {"experience": []},
            }
        }

        service = AIResumeService(provider="openai", api_key="test-key")
        result = service.generate_resume(user_info)

        assert "cv" in result
        assert result["cv"]["name"] == "John Doe"

    def test_generate_resume_empty_info(self, mock_ai_client):  # noqa: ARG002
        service = AIResumeService(provider="openai", api_key="test-key")

        with pytest.raises(RenderCVUserError, match="User information cannot be empty"):
            service.generate_resume("   ")


class TestSaveResumeYaml:
    def test_save_basic(self, tmp_path):
        resume_data = {"cv": {"name": "John Doe"}}
        output_path = tmp_path / "output.yaml"

        save_resume_yaml(resume_data, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "name: John Doe" in content
        assert "design:" in content
        assert "locale:" in content

    def test_save_with_custom_theme(self, tmp_path):
        resume_data = {"cv": {"name": "John Doe"}}
        output_path = tmp_path / "output.yaml"

        save_resume_yaml(resume_data, output_path, theme="engineering")

        content = output_path.read_text()
        assert "theme: engineering" in content

    def test_preserves_existing_design(self, tmp_path):
        resume_data = {"cv": {"name": "John Doe"}, "design": {"theme": "modernCV"}}
        output_path = tmp_path / "output.yaml"

        save_resume_yaml(resume_data, output_path, theme="classic")

        content = output_path.read_text()
        assert "theme: modernCV" in content


class TestLoadResumeYaml:
    def test_load_basic(self, tmp_path):
        yaml_file = tmp_path / "resume.yaml"
        yaml_file.write_text("cv:\n  name: John Doe\n")

        result = load_resume_yaml(yaml_file)

        assert result["cv"]["name"] == "John Doe"

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(RenderCVUserError, match="File not found"):
            load_resume_yaml(tmp_path / "nonexistent.yaml")


class TestReadJobDescription:
    def test_read_from_file(self, tmp_path):
        job_file = tmp_path / "job.txt"
        job_file.write_text("Looking for a developer...")

        result = read_job_description(str(job_file))
        assert result == "Looking for a developer..."

    def test_read_direct_text(self):
        result = read_job_description("Looking for a developer...")
        assert result == "Looking for a developer..."
