"""Locate recipe files in source and native-extension distributions."""
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


def code_asset_path(path):
    """Keep logical .py names in cache manifests, hash the actual shipped code."""
    path = Path(path)
    if path.is_file() or path.suffix != '.py':
        return path
    candidates = [path]
    # PyInstaller puts imported packages next to BrainLFNet's data directory.
    if path.parent.parent.name == 'BrainLFNet':
        candidates.append(path.parent.parent.parent / path.parent.name / path.name)
    for candidate in candidates:
        for suffix in EXTENSION_SUFFIXES:
            compiled = candidate.with_name(candidate.stem + suffix)
            if compiled.is_file():
                return compiled
    return path
