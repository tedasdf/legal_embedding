"""Configuration loading shared by thin command-line entry points."""

from pathlib import Path


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to read YAML configuration files") from exc
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value
