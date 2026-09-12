"""Create or verify the immutable evaluation snapshot of the live sample KB."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw_documents"
SOURCES = {
    REPO_ROOT / "data" / "sample_kb" / "company_hr_handbook.md": RAW_DIR
    / "company_hr_handbook.md",
    REPO_ROOT / "data" / "sample_kb" / "hr_operations_runbook.md": RAW_DIR
    / "hr_operations_runbook.md",
    REPO_ROOT / "uploads" / "HR-Policy.pdf": RAW_DIR / "HR-Policy.pdf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_snapshot() -> list[str]:
    """Return source/snapshot mismatches without changing files."""
    errors = []
    for source, destination in SOURCES.items():
        if not source.is_file():
            errors.append(f"Canonical source missing: {source}")
        elif not destination.is_file():
            errors.append(f"Evaluation snapshot missing: {destination}")
        elif _sha256(source) != _sha256(destination):
            errors.append(f"Evaluation snapshot is stale: {destination.name}")
    return errors


def sync_snapshot() -> None:
    """Copy the three canonical KB files into the versioned evaluation corpus."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for source, destination in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(f"Canonical source missing: {source}")
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or refresh rag_eval raw documents")
    parser.add_argument("--sync", action="store_true", help="Refresh snapshot from the live KB")
    args = parser.parse_args()
    if args.sync:
        sync_snapshot()
    errors = check_snapshot()
    if errors:
        raise SystemExit("\n".join(errors) + "\nRun: python -m rag_eval.prepare_data --sync")
    print("Evaluation corpus matches the two sample KB Markdown files and uploaded PDF")


if __name__ == "__main__":
    main()
