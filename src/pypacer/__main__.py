"""Main entry point for PyPaCER when run as a module."""

from .cli.pypacer_gui import main

if __name__ == "__main__":
    # Default to electrode GUI when running python -m pypacer
    main()
