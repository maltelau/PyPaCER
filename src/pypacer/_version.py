"""Version information for PyPaCER."""

# Version is read from pyproject.toml when installed
try:
    from importlib.metadata import version

    __version__ = version("pypacer")
except ImportError:
    # Fallback for Python < 3.8
    try:
        import pkg_resources

        __version__ = pkg_resources.get_distribution("pypacer").version
    except:
        # Fallback for development/editable installs
        __version__ = "0.9.0"  # Keep this synchronized with pyproject.toml
