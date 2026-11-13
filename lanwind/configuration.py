"""Configuration loading without importing a training runtime."""
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]

def config(path=None):
    return yaml.safe_load(Path(path or ROOT/'config.yaml').read_text(encoding='utf8'))
