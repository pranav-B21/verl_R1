#!/usr/bin/env python3
"""Verify immutable inputs for paper-protocol RRCM runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-validation", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["files"].items():
        path = args.data_dir / relative
        if not path.is_file():
            if relative.startswith("val_rrcm_") and not args.require_validation:
                continue
            failures.append(f"missing: {path}")
            continue
        if expected.get("sha256") is None:
            if relative.startswith("val_rrcm_") and args.require_validation:
                failures.append(f"manifest has no frozen sha256 for required validation file: {path}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            failures.append(f"sha256 mismatch: {path} expected={expected['sha256']} actual={actual_hash}")
        if "size_bytes" in expected and path.stat().st_size != expected["size_bytes"]:
            failures.append(
                f"size mismatch: {path} expected={expected['size_bytes']} actual={path.stat().st_size}"
            )
        if "lines" in expected:
            actual_lines = line_count(path)
            if actual_lines != expected["lines"]:
                failures.append(f"line mismatch: {path} expected={expected['lines']} actual={actual_lines}")
        if "records" in expected:
            records = json.loads(path.read_text(encoding="utf-8"))
            if len(records) != expected["records"]:
                failures.append(f"record mismatch: {path} expected={expected['records']} actual={len(records)}")

    if failures:
        raise SystemExit("RRCM frozen-input verification failed:\n- " + "\n- ".join(failures))
    print(f"[rrcm-manifest] verified {manifest['protocol']}")


if __name__ == "__main__":
    main()
