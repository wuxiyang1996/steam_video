"""Freeze immutable hashes for a memory-graph experiment baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def freeze_baseline(
    artifacts: dict[str, Path],
    *,
    repository: Path,
    note: str | None = None,
) -> dict[str, Any]:
    """Return a manifest without modifying any source artifact."""

    repository = repository.resolve()
    rows: list[dict[str, Any]] = []
    for name, path in sorted(artifacts.items()):
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"baseline artifact does not exist: {resolved}")
        payload = resolved.read_bytes()
        try:
            display_path = str(resolved.relative_to(repository))
        except ValueError:
            display_path = str(resolved)
        rows.append(
            {
                "name": name,
                "path": display_path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema_version": "steam-memory-graph-baseline/v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(repository),
        "git_commit": _git(repository, "rev-parse", "HEAD"),
        "git_branch": _git(repository, "branch", "--show-current"),
        "note": note,
        "artifacts": rows,
    }


def verify_baseline(
    manifest: dict[str, Any],
    *,
    repository: Path,
) -> dict[str, Any]:
    """Verify current files against a frozen manifest."""

    repository = repository.resolve()
    checks: list[dict[str, Any]] = []
    for row in manifest.get("artifacts") or []:
        path = Path(str(row.get("path") or ""))
        resolved = path if path.is_absolute() else repository / path
        if not resolved.is_file():
            checks.append(
                {"name": row.get("name"), "path": str(path), "passed": False, "reason": "missing"}
            )
            continue
        payload = resolved.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        checks.append(
            {
                "name": row.get("name"),
                "path": str(path),
                "passed": actual == row.get("sha256") and len(payload) == row.get("size_bytes"),
                "expected_sha256": row.get("sha256"),
                "actual_sha256": actual,
                "expected_size_bytes": row.get("size_bytes"),
                "actual_size_bytes": len(payload),
            }
        )
    return {
        "schema_version": "steam-memory-graph-baseline-verification/v0.1",
        "manifest_schema_version": manifest.get("schema_version"),
        "artifact_count": len(checks),
        "passed": bool(checks) and all(row["passed"] for row in checks),
        "checks": checks,
    }


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _artifact(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("artifact must use NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", action="append", type=_artifact)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--note")
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.verify_manifest:
        manifest = json.loads(args.verify_manifest.read_text(encoding="utf-8"))
        report = verify_baseline(manifest, repository=args.repository)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    if not args.artifact or args.output is None:
        parser.error("freeze mode requires --artifact and --output")
    artifacts = dict(args.artifact)
    if len(artifacts) != len(args.artifact):
        raise ValueError("artifact names must be unique")
    manifest = freeze_baseline(
        artifacts,
        repository=args.repository,
        note=args.note,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact_count": len(artifacts), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
