"""AI-powered resume commands for RenderCV."""

import pathlib
from typing import Annotated

import rich.panel
import rich.progress
import typer
from rich import print

from rendercv.ai.service import (
    AIResumeService,
    load_resume_yaml,
    read_job_description,
    save_resume_yaml,
)
from rendercv.schema.models.design.built_in_design import available_themes
from rendercv.schema.models.locale.locale import available_locales

from ..app import app
from ..error_handler import handle_user_errors

# Create a sub-app for AI commands
ai_app = typer.Typer(
    name="ai",
    help="AI-powered resume generation and improvement commands.",
    rich_markup_mode="rich",
)
app.add_typer(ai_app)


# Common options for AI commands
def provider_option() -> Annotated[
    str | None,
    typer.Option(
        "--provider",
        "-p",
        help="AI provider: 'openai' or 'anthropic'. Auto-detects from API key if not"
        " specified.",
    ),
]:
    return None


def api_key_option() -> Annotated[
    str | None,
    typer.Option(
        "--api-key",
        "-k",
        help="API key for the AI provider. Can also use OPENAI_API_KEY or"
        " ANTHROPIC_API_KEY environment variables.",
        envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    ),
]:
    return None


def model_option() -> Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Model to use. Defaults: gpt-4o (OpenAI), claude-sonnet-4-20250514 (Anthropic).",
    ),
]:
    return None


@ai_app.command(
    name="parse",
    help=(
        "Parse an existing resume (PDF, DOCX, TXT) into RenderCV YAML format."
        " Example: [yellow]rendercv ai parse resume.pdf[/yellow]"
    ),
)
@handle_user_errors
def ai_parse(
    input_file: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to the resume file (PDF, DOCX, or TXT)",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output YAML file path. Defaults to <input_name>_CV.yaml",
        ),
    ] = None,
    theme: Annotated[
        str,
        typer.Option(
            "--theme",
            "-t",
            help=f"Theme for the generated CV (available: {', '.join(available_themes)})",
        ),
    ] = "classic",
    locale: Annotated[
        str,
        typer.Option(
            "--locale",
            "-l",
            help=f"Locale for the generated CV (available: {', '.join(available_locales)})",
        ),
    ] = "english",
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: 'openai' or 'anthropic'",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="API key for the AI provider",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to use",
        ),
    ] = None,
):
    """Parse an existing resume into RenderCV YAML format using AI."""
    # Determine output path
    if output is None:
        output = pathlib.Path(f"{input_file.stem}_CV.yaml")

    print(f"\n[bold]Parsing resume:[/bold] {input_file}")

    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Extracting and parsing resume with AI...", total=None)

        # Initialize AI service and parse
        service = AIResumeService(provider=provider, api_key=api_key, model=model)
        resume_data = service.parse_resume(input_file)

        # Save to YAML
        save_resume_yaml(resume_data, output, theme=theme, locale=locale)

    print(
        rich.panel.Panel(
            f"[green]✓[/green] Successfully parsed resume to:"
            f" [purple]{output}[/purple]\n\n"
            "Next steps:\n"
            f"  1. Review and edit the YAML file\n"
            f"  2. Run: [cyan]rendercv render {output}[/cyan]\n\n"
            "Optional: Polish the resume with:\n"
            f"  [cyan]rendercv ai polish {output}[/cyan]",
            title="Resume Parsed",
            title_align="left",
            border_style="green",
        )
    )


@ai_app.command(
    name="polish",
    help=(
        "Polish and improve resume content using AI."
        " Example: [yellow]rendercv ai polish John_Doe_CV.yaml[/yellow]"
    ),
)
@handle_user_errors
def ai_polish(
    input_file: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to the RenderCV YAML file",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output YAML file path. Defaults to <input>_polished.yaml",
        ),
    ] = None,
    focus: Annotated[
        list[str] | None,
        typer.Option(
            "--focus",
            "-f",
            help="Areas to focus improvement on (can specify multiple)",
        ),
    ] = None,
    in_place: Annotated[
        bool,
        typer.Option(
            "--in-place",
            "-i",
            help="Modify the input file in place",
        ),
    ] = False,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: 'openai' or 'anthropic'",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="API key for the AI provider",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to use",
        ),
    ] = None,
):
    """Polish and improve resume content using AI."""
    # Determine output path
    if in_place:
        output = input_file
    elif output is None:
        output = pathlib.Path(f"{input_file.stem}_polished.yaml")

    print(f"\n[bold]Polishing resume:[/bold] {input_file}")
    if focus:
        print(f"[bold]Focus areas:[/bold] {', '.join(focus)}")

    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Improving resume content with AI...", total=None)

        # Load existing resume
        resume_data = load_resume_yaml(input_file)

        # Initialize AI service and polish
        service = AIResumeService(provider=provider, api_key=api_key, model=model)
        polished_data = service.polish_resume(resume_data, focus_areas=focus)

        # Preserve design and locale from original if present
        if "design" in resume_data and "design" not in polished_data:
            polished_data["design"] = resume_data["design"]
        if "locale" in resume_data and "locale" not in polished_data:
            polished_data["locale"] = resume_data["locale"]
        if "settings" in resume_data and "settings" not in polished_data:
            polished_data["settings"] = resume_data["settings"]

        # Save polished resume
        save_resume_yaml(polished_data, output)

    action = "Updated" if in_place else "Created polished resume at"
    print(
        rich.panel.Panel(
            f"[green]✓[/green] {action}: [purple]{output}[/purple]\n\n"
            "The AI has improved:\n"
            "  • Action verbs and impact statements\n"
            "  • Quantifiable metrics where possible\n"
            "  • Professional language and clarity\n\n"
            f"Next: [cyan]rendercv render {output}[/cyan]",
            title="Resume Polished",
            title_align="left",
            border_style="green",
        )
    )


@ai_app.command(
    name="tailor",
    help=(
        "Tailor a resume for a specific job description."
        " Example: [yellow]rendercv ai tailor cv.yaml --job job.txt[/yellow]"
    ),
)
@handle_user_errors
def ai_tailor(
    input_file: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to the RenderCV YAML file",
            exists=True,
            readable=True,
        ),
    ],
    job: Annotated[
        str,
        typer.Option(
            "--job",
            "-j",
            help="Job description (file path or direct text)",
        ),
    ],
    output: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output YAML file path. Defaults to <input>_tailored.yaml",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: 'openai' or 'anthropic'",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="API key for the AI provider",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to use",
        ),
    ] = None,
):
    """Tailor a resume to match a specific job description using AI."""
    # Determine output path
    if output is None:
        output = pathlib.Path(f"{input_file.stem}_tailored.yaml")

    # Read job description
    job_description = read_job_description(job)

    print(f"\n[bold]Tailoring resume:[/bold] {input_file}")
    print(f"[bold]Job description:[/bold] {len(job_description)} characters")

    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Tailoring resume to job description with AI...", total=None)

        # Load existing resume
        resume_data = load_resume_yaml(input_file)

        # Initialize AI service and tailor
        service = AIResumeService(provider=provider, api_key=api_key, model=model)
        tailored_data = service.tailor_resume(resume_data, job_description)

        # Preserve design and locale from original
        if "design" in resume_data and "design" not in tailored_data:
            tailored_data["design"] = resume_data["design"]
        if "locale" in resume_data and "locale" not in tailored_data:
            tailored_data["locale"] = resume_data["locale"]
        if "settings" in resume_data and "settings" not in tailored_data:
            tailored_data["settings"] = resume_data["settings"]

        # Save tailored resume
        save_resume_yaml(tailored_data, output)

    print(
        rich.panel.Panel(
            f"[green]✓[/green] Created tailored resume: [purple]{output}[/purple]\n\n"
            "The AI has optimized your resume by:\n"
            "  • Aligning keywords with job requirements\n"
            "  • Prioritizing relevant experience\n"
            "  • Emphasizing matching skills\n"
            "  • Optimizing for ATS compatibility\n\n"
            f"Next: [cyan]rendercv render {output}[/cyan]",
            title="Resume Tailored",
            title_align="left",
            border_style="green",
        )
    )


@ai_app.command(
    name="generate",
    help=(
        "Generate a new resume from scratch using AI."
        " Example: [yellow]rendercv ai generate --name 'John Doe'[/yellow]"
    ),
)
@handle_user_errors
def ai_generate(
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Your full name",
        ),
    ],
    info: Annotated[
        str | None,
        typer.Option(
            "--info",
            "-i",
            help="Path to a file with your background info, or direct text",
        ),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            help="Interactive mode: prompts for information",
        ),
    ] = False,
    output: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output YAML file path. Defaults to <name>_CV.yaml",
        ),
    ] = None,
    theme: Annotated[
        str,
        typer.Option(
            "--theme",
            "-t",
            help=f"Theme for the generated CV (available: {', '.join(available_themes)})",
        ),
    ] = "classic",
    locale: Annotated[
        str,
        typer.Option(
            "--locale",
            "-l",
            help=f"Locale for the generated CV (available: {', '.join(available_locales)})",
        ),
    ] = "english",
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="AI provider: 'openai' or 'anthropic'",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="API key for the AI provider",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to use",
        ),
    ] = None,
):
    """Generate a new resume from scratch using AI."""
    # Determine output path
    if output is None:
        output = pathlib.Path(f"{name.replace(' ', '_')}_CV.yaml")

    # Gather user information
    user_info = f"Name: {name}\n\n"

    if interactive:
        print("\n[bold]Interactive Resume Generation[/bold]")
        print("Please provide information about yourself (press Enter twice when done):\n")

        # Gather information interactively
        sections = [
            ("Current or desired job title/headline", "headline"),
            ("Location (City, State/Country)", "location"),
            ("Email address", "email"),
            ("Phone number (with country code)", "phone"),
            ("LinkedIn username (optional)", "linkedin"),
            ("GitHub username (optional)", "github"),
            ("Website URL (optional)", "website"),
            (
                "Education (include school, degree, field, dates, achievements)",
                "education",
            ),
            (
                "Work experience (include company, title, dates, responsibilities)",
                "experience",
            ),
            ("Skills (technical and soft skills)", "skills"),
            ("Projects (optional)", "projects"),
            ("Certifications/Awards (optional)", "certifications"),
        ]

        for prompt_text, key in sections:
            print(f"\n[cyan]{prompt_text}:[/cyan]")
            lines = []
            while True:
                line = input()
                if line == "":
                    if lines and lines[-1] == "":
                        break
                    lines.append(line)
                else:
                    lines.append(line)
            value = "\n".join(lines).strip()
            if value:
                user_info += f"\n{key.title()}:\n{value}\n"

    elif info:
        # Read from file or use as text
        info_text = read_job_description(info)  # Reuse file reading logic
        user_info += info_text
    else:
        print("\n[bold]Tip:[/bold] For better results, provide your background info:")
        print("  • Use [cyan]--info[/cyan] with a file path or text")
        print("  • Or use [cyan]--interactive[/cyan] for guided input\n")
        user_info += (
            "Please generate a sample professional resume for a software engineer "
            "with 5 years of experience."
        )

    print(f"\n[bold]Generating resume for:[/bold] {name}")

    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Generating resume with AI...", total=None)

        # Initialize AI service and generate
        service = AIResumeService(provider=provider, api_key=api_key, model=model)
        resume_data = service.generate_resume(user_info)

        # Save to YAML
        save_resume_yaml(resume_data, output, theme=theme, locale=locale)

    print(
        rich.panel.Panel(
            f"[green]✓[/green] Generated resume: [purple]{output}[/purple]\n\n"
            "Next steps:\n"
            f"  1. Review and customize the generated content\n"
            f"  2. Run: [cyan]rendercv render {output}[/cyan]\n\n"
            "Optional improvements:\n"
            f"  • Polish: [cyan]rendercv ai polish {output}[/cyan]\n"
            f"  • Tailor for a job: [cyan]rendercv ai tailor {output} --job <description>[/cyan]",
            title="Resume Generated",
            title_align="left",
            border_style="green",
        )
    )
