"""Command line entry point."""

import typer

from parity import __version__

app = typer.Typer(help="Statistical conformance testing for LLM inference endpoints.")


@app.callback()
def main() -> None:
    """Parity command group."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)
