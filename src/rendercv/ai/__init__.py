"""AI-powered resume generation and improvement features for RenderCV."""

from .client import AIClient, get_ai_client
from .parser import extract_resume_text
from .prompts import (
    GENERATE_RESUME_PROMPT,
    PARSE_RESUME_PROMPT,
    POLISH_RESUME_PROMPT,
    TAILOR_RESUME_PROMPT,
)
from .service import (
    AIResumeService,
    load_resume_yaml,
    read_job_description,
    save_resume_yaml,
)

__all__ = [
    "GENERATE_RESUME_PROMPT",
    "PARSE_RESUME_PROMPT",
    "POLISH_RESUME_PROMPT",
    "TAILOR_RESUME_PROMPT",
    "AIClient",
    "AIResumeService",
    "extract_resume_text",
    "get_ai_client",
    "load_resume_yaml",
    "read_job_description",
    "save_resume_yaml",
]
