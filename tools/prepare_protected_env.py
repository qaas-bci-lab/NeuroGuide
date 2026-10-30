"""Create an isolated Cython build overlay using the selected runtime's dependencies.

Invoke this script with the Python interpreter whose CUDA/PyTorch stack should
be shipped. This never installs packages into that runtime environment.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=root / 'build/protected_V4_5')
    args = parser.parse_args()
    stage = args.stage.resolve()
    target = stage / 'venv'
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if Path(sys.prefix).resolve() == target:
        raise RuntimeError('Run preparation with your original runtime interpreter, not the build overlay.')
    env = os.environ.copy()
    for key, folder in (('TEMP', 'temp'), ('TMP', 'temp'), ('PIP_CACHE_DIR', 'pip_cache')):
        env[key] = str(stage / folder)
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(target)], env=env, check=True)
    sites = []
    for value in sys.path:
        path = Path(value)
        if path.name.casefold() == 'site-packages' and path.is_dir() and str(path) not in sites:
            sites.append(str(path))
    site = target / ('Lib/site-packages' if os.name == 'nt' else f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    (site / 'neuroguide_dependencies.pth').write_text(
        'import sys; _ng_index = [p.casefold() for p in sys.path].index(' + repr(str(site).casefold()) + ') + 1; '
        'sys.path[_ng_index:_ng_index] = ' + repr(sites) + '\n', encoding='utf-8')
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '-r', str(root / 'requirements-build-protected.txt')], env=env, check=True)
    print('BUILD INTERPRETER:', python)


if __name__ == '__main__':
    main()
