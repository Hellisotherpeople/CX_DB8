"""Rich-powered terminal display for CX_DB8."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from cx_db8.summarizer import SummaryResult

THEME_HIGHLIGHT = "bold yellow on grey23"
THEME_UNDERLINE = "underline bright_white"
THEME_REMOVED = "dim"
THEME_ACCENT = "bold cyan"
THEME_TITLE = "bold magenta"


def render_summary(result: SummaryResult, console: Console | None = None) -> Text:
    """Render a summarized card as Rich Text with highlights and underlines."""
    text = Text()

    if result.granularity.value in ("phrase", "word"):
        for span in result.spans:
            if span.score >= result.highlight_threshold:
                text.append(span.text, style=THEME_HIGHLIGHT)
            elif span.score >= result.underline_threshold:
                text.append(span.text, style=THEME_UNDERLINE)
            else:
                text.append(span.text, style=THEME_REMOVED)
            text.append(" ")
    else:
        for span in result.spans:
            if span.score >= result.highlight_threshold:
                text.append(span.text, style=THEME_HIGHLIGHT)
            elif span.score >= result.underline_threshold:
                text.append(span.text, style=THEME_UNDERLINE)
            else:
                text.append(span.text, style=THEME_REMOVED)
            text.append("\n\n")

    return text


def print_summary(
    result: SummaryResult,
    author: str = "",
    citation: str = "",
    console: Console | None = None,
) -> None:
    """Print a full summary display to the console."""
    if console is None:
        console = Console()

    # Header
    console.print()
    console.rule("[bold magenta]CX_DB8 Summary[/]", style="magenta")
    console.print()

    # Metadata table
    meta = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 2))
    meta.add_column("Key", style=THEME_ACCENT, min_width=12)
    meta.add_column("Value")
    meta.add_row("Query", escape(result.query[:120]))
    meta.add_row("Granularity", result.granularity.value.capitalize())
    meta.add_row("Underline", f"top {100 - result.underline_percentile:.0f}%")
    meta.add_row("Highlight", f"top {100 - result.highlight_percentile:.0f}%")
    if author:
        meta.add_row("Author", escape(author))
    if citation:
        meta.add_row("Citation", escape(citation))
    console.print(meta)
    console.print()

    # Summary panel
    summary_text = render_summary(result, console)
    console.print(
        Panel(
            summary_text,
            title="[bold]Summary[/]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Stats
    stats = Table(box=box.ROUNDED, title="Statistics", title_style="bold cyan")
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", justify="right", style="bold")
    stats.add_row("Total spans", str(len(result.spans)))
    stats.add_row("Highlighted", f"[bold yellow]{len(result.highlighted)}[/]")
    stats.add_row("Underlined", f"[white]{len(result.underlined)}[/]")
    stats.add_row("Removed", f"[dim]{len(result.removed)}[/]")
    stats.add_row(
        "Compression",
        f"{result.compression_ratio:.1%}",
    )
    console.print(stats)
    console.print()


def export_svg(result: SummaryResult, path: Path, author: str = "", citation: str = "") -> Path:
    """Export summary as an SVG file using Rich's SVG export."""
    svg_console = Console(file=StringIO(), width=100, record=True)
    print_summary(result, author=author, citation=citation, console=svg_console)
    svg_text = svg_console.export_svg(title="CX_DB8 Summary")
    path.write_text(svg_text)
    return path


def export_html(result: SummaryResult, path: Path, author: str = "", citation: str = "") -> Path:
    """Export summary as an HTML file using Rich's HTML export."""
    html_console = Console(file=StringIO(), width=100, record=True)
    print_summary(result, author=author, citation=citation, console=html_console)
    html_text = html_console.export_html()
    path.write_text(html_text)
    return path
