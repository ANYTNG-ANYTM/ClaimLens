"""Project paths and configuration."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def path(relative):
    return ROOT / relative


def load_config():
    with path('config/config.yaml').open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)
