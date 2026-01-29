"""AI service for resume operations: parse, polish, tailor, and generate."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

import ruamel.yaml

from rendercv.exception import RenderCVUserError

from .client import AIClient, get_ai_client
from .parser import extract_resume_text
from .prompts import (
    GENERATE_RESUME_PROMPT,
    GENERATE_RESUME_USER_PROMPT,
    PARSE_RESUME_PROMPT,
    PARSE_RESUME_USER_PROMPT,
    POLISH_RESUME_PROMPT,
    POLISH_RESUME_USER_PROMPT,
    TAILOR_RESUME_PROMPT,
    TAILOR_RESUME_USER_PROMPT,
)

if TYPE_CHECKING:
    pass


class AIResumeService:
    """Service class for AI-powered resume operations."""

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the AI resume service.

        Args:
            provider: AI provider ("openai" or "anthropic").
            api_key: API key for the provider.
            model: Model to use (defaults based on provider).
        """
        self.client: AIClient = get_ai_client(
            provider=provider, api_key=api_key, model=model
        )

    def parse_resume(self, file_path: pathlib.Path) -> dict[str, Any]:
        """Parse an existing resume file into RenderCV format.

        Args:
            file_path: Path to the resume file (PDF, DOCX, or TXT).

        Returns:
            Parsed resume data in RenderCV format.
        """
        # Extract text from the file
        resume_text = extract_resume_text(file_path)

        if not resume_text.strip():
            message = (
                f"Could not extract any text from {file_path}. The file may be"
                " image-based or corrupted."
            )
            raise RenderCVUserError(message)

        # Use AI to parse the resume
        prompt = PARSE_RESUME_USER_PROMPT.format(resume_text=resume_text)
        result = self.client.generate_json(
            prompt=prompt,
            system_prompt=PARSE_RESUME_PROMPT,
            temperature=0.2,  # Lower temperature for more consistent parsing
        )

        # Validate basic structure
        if "cv" not in result:
            raise RenderCVUserError(
                "AI failed to generate valid resume structure. Please try again."
            )

        return result

    def polish_resume(
        self, resume_data: dict[str, Any], focus_areas: list[str] | None = None
    ) -> dict[str, Any]:
        """Polish and improve resume content.

        Args:
            resume_data: Resume data in RenderCV format.
            focus_areas: Optional list of areas to focus improvement on.

        Returns:
            Polished resume data.
        """
        resume_json = json.dumps(resume_data, indent=2)

        # Build prompt with optional focus areas
        prompt = POLISH_RESUME_USER_PROMPT.format(resume_json=resume_json)
        if focus_areas:
            focus_str = ", ".join(focus_areas)
            prompt += f"\n\nPlease focus especially on improving: {focus_str}"

        result = self.client.generate_json(
            prompt=prompt,
            system_prompt=POLISH_RESUME_PROMPT,
            temperature=0.5,
        )

        if "cv" not in result:
            raise RenderCVUserError(
                "AI failed to generate valid polished resume. Please try again."
            )

        return result

    def tailor_resume(
        self, resume_data: dict[str, Any], job_description: str
    ) -> dict[str, Any]:
        """Tailor resume to a specific job description.

        Args:
            resume_data: Resume data in RenderCV format.
            job_description: The job description text to tailor for.

        Returns:
            Tailored resume data.
        """
        if not job_description.strip():
            raise RenderCVUserError("Job description cannot be empty.")

        resume_json = json.dumps(resume_data, indent=2)
        prompt = TAILOR_RESUME_USER_PROMPT.format(
            job_description=job_description, resume_json=resume_json
        )

        result = self.client.generate_json(
            prompt=prompt,
            system_prompt=TAILOR_RESUME_PROMPT,
            temperature=0.4,
        )

        # Extract the resume part (may include analysis)
        if "cv" not in result:
            # Check if it's nested under 'resume' key
            if "resume" in result and "cv" in result["resume"]:
                result = result["resume"]
            else:
                raise RenderCVUserError(
                    "AI failed to generate valid tailored resume. Please try again."
                )

        return result

    def generate_resume(self, user_info: str) -> dict[str, Any]:
        """Generate a new resume from user-provided information.

        Args:
            user_info: Free-form text with user's background information.

        Returns:
            Generated resume data in RenderCV format.
        """
        if not user_info.strip():
            raise RenderCVUserError("User information cannot be empty.")

        prompt = GENERATE_RESUME_USER_PROMPT.format(user_info=user_info)

        result = self.client.generate_json(
            prompt=prompt,
            system_prompt=GENERATE_RESUME_PROMPT,
            temperature=0.6,
        )

        if "cv" not in result:
            raise RenderCVUserError(
                "AI failed to generate valid resume. Please try again."
            )

        return result


def save_resume_yaml(
    resume_data: dict[str, Any],
    output_path: pathlib.Path,
    theme: str = "classic",
    locale: str = "english",
) -> None:
    """Save resume data to a YAML file.

    Args:
        resume_data: Resume data dictionary.
        output_path: Path to save the YAML file.
        theme: Theme name for the design section.
        locale: Locale name for translations.
    """
    # Add design and locale sections if not present
    if "design" not in resume_data:
        resume_data["design"] = {"theme": theme}
    if "locale" not in resume_data:
        resume_data["locale"] = {"language": locale}

    # Use ruamel.yaml for nice formatting
    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(resume_data, f)


def load_resume_yaml(file_path: pathlib.Path) -> dict[str, Any]:
    """Load resume data from a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Resume data dictionary.
    """
    if not file_path.exists():
        message = f"File not found: {file_path}"
        raise RenderCVUserError(message)

    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.load(f)
        if not isinstance(data, dict):
            message = f"Invalid YAML structure in {file_path}"
            raise RenderCVUserError(message)
        return dict(data)
    except RenderCVUserError:
        raise
    except Exception as e:
        message = f"Failed to load YAML file: {e}"
        raise RenderCVUserError(message) from e


def read_job_description(source: str) -> str:
    """Read job description from a file path or return as-is if it's text.

    Args:
        source: Either a file path or the job description text itself.

    Returns:
        The job description text.
    """
    # Check if it's a file path
    path = pathlib.Path(source)
    if path.exists() and path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            pass  # Fall through to return as-is

    # Return as-is (it's direct text)
    return source
