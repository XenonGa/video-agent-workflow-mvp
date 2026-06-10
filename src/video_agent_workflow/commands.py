from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run_template_command(template: str, values: dict[str, object]) -> None:
    command = template.format(**{key: _quote(value) for key, value in values.items()})
    subprocess.run(command, shell=True, check=True)


def _quote(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(shlex.quote(str(item)) for item in value)
    return shlex.quote(str(value))


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} was not created: {path}")
    return path
