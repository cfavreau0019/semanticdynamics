"""
File-backed store for one generation run: data/generations/<run_id>/<table>.jsonl.

Append-only tables are appended to (crash-safe: a partially written run keeps every row
written so far); mutable tables (runs, batches) are small and rewritten atomically.
Every row is checked against data_generation.schema before it is written.
"""
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from data_generation.schema import TABLES, check_row

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "generations"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def code_version() -> tuple[Optional[str], Optional[bool]]:
    """(git commit, has uncommitted changes) of the repo, or (None, None) outside git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return sha, bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return None, None


class RunStore:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.run_id = self.dir.name

    @classmethod
    def create(cls, root: str | Path = DEFAULT_ROOT, run_id: Optional[str] = None) -> "RunStore":
        run_dir = Path(root) / (run_id or new_run_id())
        run_dir.mkdir(parents=True, exist_ok=False)
        return cls(run_dir)

    @classmethod
    def open(cls, run_id: str, root: str | Path = DEFAULT_ROOT) -> "RunStore":
        run_dir = Path(root) / run_id
        if not (run_dir / "runs.jsonl").exists():
            raise FileNotFoundError(f"No run at {run_dir}")
        return cls(run_dir)

    def path(self, table: str) -> Path:
        if table not in TABLES:
            raise KeyError(f"Unknown table {table!r}")
        return self.dir / f"{table}.jsonl"

    @property
    def batch_files_dir(self) -> Path:
        return self.dir / "batch_files"

    # ---- writing -------------------------------------------------------------------
    def append(self, table: str, rows: Iterable[dict[str, Any]]) -> int:
        if TABLES[table].mutable:
            raise ValueError(f"{table} is mutable; use upsert()")
        checked = [check_row(table, r) for r in rows]
        if checked:
            with open(self.path(table), "a", encoding="utf-8") as f:
                for r in checked:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return len(checked)

    def upsert(self, table: str, rows: Iterable[dict[str, Any]]) -> None:
        """Insert or replace rows by primary key, rewriting the (small) file atomically."""
        t = TABLES[table]
        if not t.mutable:
            raise ValueError(f"{table} is append-only; use append()")
        key = lambda r: tuple(r[k] for k in t.primary_key)
        current = {key(r): r for r in self.read(table)}
        for r in rows:
            r = check_row(table, r)
            current[key(r)] = r
        tmp = self.path(table).with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in current.values():
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, self.path(table))

    # ---- reading -------------------------------------------------------------------
    def read(self, table: str) -> list[dict[str, Any]]:
        p = self.path(table)
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def run(self) -> dict[str, Any]:
        rows = self.read("runs")
        if not rows:
            raise ValueError(f"{self.dir} has no runs row")
        return rows[0]

    def update_run(self, **changes) -> dict[str, Any]:
        row = {**self.run(), **changes, "updated_at": utc_now()}
        self.upsert("runs", [row])
        return row
