#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys


SYNC_SOURCES = [
    "src/model_explorer_onnx/main.py",
    "src/model_explorer_onnx/_passes.py",
]
TARGET = "web/src/python/convert.py"
STAMP_PREFIX = "# SOURCE_SYNC_SHA256: "


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _compute_digest(root: pathlib.Path) -> str:
    h = hashlib.sha256()
    for rel in SYNC_SOURCES:
        path = root / rel
        data = path.read_bytes()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()


def _update_stamp(content: str, digest: str) -> str:
    stamp_line = f"{STAMP_PREFIX}{digest}"
    if STAMP_PREFIX in content:
        return re.sub(
            rf"^{re.escape(STAMP_PREFIX)}[0-9a-f]+$",
            stamp_line,
            content,
            flags=re.MULTILINE,
        )
    return (
        "# NOTE: This file is web-adapted and maintained alongside source adapter.\n"
        f"# SOURCE_SYNC_FILES: {', '.join(SYNC_SOURCES)}\n"
        f"{stamp_line}\n\n{content}"
    )


def _read_stored_stamp(content: str) -> str | None:
    for line in content.splitlines():
        if line.startswith(STAMP_PREFIX):
            return line[len(STAMP_PREFIX) :].strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["check", "update"],
        default="check",
        help="check: fail if stamp mismatch; update: rewrite stamp in target file",
    )
    args = parser.parse_args()

    root = _repo_root()
    target = root / TARGET
    content = target.read_text(encoding="utf-8")
    digest = _compute_digest(root)

    if args.mode == "update":
        target.write_text(_update_stamp(content, digest), encoding="utf-8")
        print(f"Updated sync stamp in {TARGET}: {digest}")
        return 0

    stored = _read_stored_stamp(content)
    if stored != digest:
        print(
            "web converter is out of sync with source adapter files.\n"
            f"Expected: {digest}\n"
            f"Found:    {stored or '<missing>'}\n\n"
            "After updating web/src/python/convert.py, run:\n"
            "  python tools/web/sync_web_converter.py --mode update",
            file=sys.stderr,
        )
        return 1
    print("web converter sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
