"""Main entry point for PyPaCER CLI."""

import click

from . import pypacer, pypacer_gpu, pypacer_gui, pypacer_report


@click.group()
@click.version_option()
def cli():
    """PyPaCER"""
    pass


@cli.command()
@click.argument("args", nargs=-1)
def report(args):
    """Generate electrode reconstruction reports."""
    pypacer_report.main(list(args))


@cli.command()
@click.argument("args", nargs=-1)
def gui(args):
    """Launch the electrode GUI."""
    pypacer_gui.main(list(args))


@cli.command()
@click.argument("args", nargs=-1)
def gpu(args):
    """Run GPU-accelerated electrode reconstruction."""
    pypacer_gpu.main(list(args))


@cli.command(name="run")
@click.argument("args", nargs=-1)
def run_cpu(args):
    """Run CPU-based electrode reconstruction (default)."""
    pypacer.main(list(args))


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
