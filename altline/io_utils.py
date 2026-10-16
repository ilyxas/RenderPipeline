from __future__ import annotations

import json
from pathlib import Path

ALTLINE_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ALTLINE_ROOT / "runs"
REPORTS_ROOT = ALTLINE_ROOT / "reports"


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
