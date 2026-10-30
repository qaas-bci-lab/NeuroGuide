"""Resolve losslessly shared template files while preserving logical filenames."""
import json
from pathlib import Path


def template_files(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    index = root / '.resource_index.json'
    if not index.is_file():
        return sorted(p.name for p in directory.iterdir())
    files = json.loads(index.read_text(encoding='utf-8'))['files']
    parent = directory.relative_to(root).as_posix()
    return sorted(Path(name).name for name in files if Path(name).parent.as_posix() == parent)


def template_path(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    index = root / '.resource_index.json'
    if not index.is_file():
        return str(path)
    relative = path.relative_to(root).as_posix()
    files = json.loads(index.read_text(encoding='utf-8'))['files']
    resolved = (root / files.get(relative, relative)).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Template resource is outside the package')
    return str(resolved)
