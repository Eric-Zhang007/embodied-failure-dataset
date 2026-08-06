"""Build and verify the anonymous, code-only review supplement."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT_FILES = ("pyproject.toml", "uv.lock")
CORE_SCRIPTS = (
    "alfred_replay.py",
    "analyze_ablation.py",
    "analyze_collection_snapshot.py",
    "audit_historical_outputs.py",
    "build_paper_figures.py",
    "download_alfred.py",
    "e2e_test.py",
    "generate_ft_data.py",
    "replay_demo.py",
    "resume_episode.py",
    "run_pipeline.py",
    "run_timed_ablation.py",
)
GENERATED_FILES = {
    "README.md": """# Anonymous Code Supplement

This archive contains the code needed to run the collection pipeline, replay
recorded episodes, and reproduce the diagnostic snapshot reported in the paper.
It intentionally excludes raw outputs, tests, development notes, and paper
sources.

## Environment

- Linux or WSL2
- Python 3.10 or later
- AI2-THOR 5.0.0
- ALFRED `json_2.1.0` task data under `data/json_2.1.0/` (not bundled)

Install the locked environment with `uv sync`. Copy `config.example.toml` to
`config.toml`, fill in the model endpoint and credentials, then run:

```bash
uv run python scripts/run_pipeline.py --config config.toml --max 1 --output output_review
```

To recompute the frozen-snapshot statistics from a supplied collector directory:

```bash
uv run python scripts/analyze_collection_snapshot.py <collector-directory> \
  --max-depth 3 --output collection_snapshot_stats.json
```

The full collector snapshot is not bundled in this code-only archive.

To run the matched component pilot used in the paper:

```bash
uv run python scripts/run_timed_ablation.py --config config.toml
```
""",
    "config.example.toml": """[api]
base_url = ""
api_key = ""

[models]
planner = ""
executor = ""
oracle = ""

[reasoning_effort]
planner = "high"
executor = "medium"
oracle = "high"
""",
}

_FORBIDDEN_PARTS = {
    ".git",
    ".planning",
    ".pytest_cache",
    "authorkit27",
    "deprecated",
    "docs",
    "logs",
    "paper",
    "tests",
    "work",
}
_FORBIDDEN_FILENAMES = {
    "claude.md",
    "config.toml",
    "design.md",
    "planner_executor_research.md",
}
_IDENTITY_PATTERNS = (
    re.compile(rb"Eric-Zhang007", re.IGNORECASE),
    re.compile(rb"/home/zjc", re.IGNORECASE),
    re.compile(rb"Users[\\/]zjc", re.IGNORECASE),
    re.compile(rb"(?:api[_-]?key\s*[=:]\s*[\"']?)sk-[A-Za-z0-9_-]{16,}", re.IGNORECASE),
)
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def _allowed_path(name: str) -> bool:
    path = PurePosixPath(name)
    lowered = tuple(part.lower() for part in path.parts)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return False
    if any(part in _FORBIDDEN_PARTS or part.startswith("output") for part in lowered):
        return False
    if lowered[-1] in _FORBIDDEN_FILENAMES:
        return False
    if name in {*ROOT_FILES, *GENERATED_FILES}:
        return True
    if len(path.parts) == 2 and path.parts[0] == "src" and path.suffix == ".py":
        return True
    return (
        len(path.parts) == 2
        and path.parts[0] == "scripts"
        and path.name in CORE_SCRIPTS
    )


def _source_files(source: Path) -> dict[str, bytes]:
    files = {name: (source / name).read_bytes() for name in ROOT_FILES}
    src_dir = source / "src"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source package: {src_dir}")
    for path in sorted(src_dir.glob("*.py")):
        files[path.relative_to(source).as_posix()] = path.read_bytes()
    for name in CORE_SCRIPTS:
        path = source / "scripts" / name
        if path.is_file():
            files[path.relative_to(source).as_posix()] = path.read_bytes()
    files.update({name: text.encode("utf-8") for name, text in GENERATED_FILES.items()})
    return files


def _check_entry(name: str, payload: bytes) -> None:
    if not _allowed_path(name):
        raise ValueError(f"Forbidden or non-allowlisted supplement path: {name}")
    for pattern in _IDENTITY_PATTERNS:
        if pattern.search(payload):
            raise ValueError(f"Identity marker found in anonymous supplement file: {name}")


def verify_code_supplement(archive: Path) -> dict[str, int | str]:
    archive = Path(archive)
    seen: set[str] = set()
    with ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            if info.filename in seen:
                raise ValueError(f"Duplicate archive path: {info.filename}")
            seen.add(info.filename)
            _check_entry(info.filename, bundle.read(info))
    required = {*ROOT_FILES, *GENERATED_FILES, "scripts/run_pipeline.py"}
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"Supplement is missing required files: {missing}")
    return {
        "file_count": len(seen),
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }


def build_code_supplement(source: Path, archive: Path) -> dict[str, int | str]:
    source = Path(source).resolve()
    archive = Path(archive).resolve()
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)

    files = _source_files(source)
    for name, payload in files.items():
        _check_entry(name, payload)
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(files):
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, files[name])
    return verify_code_supplement(archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        parser.error("provide exactly one of --output or --verify")
    report = (
        build_code_supplement(args.source, args.output)
        if args.output
        else verify_code_supplement(args.verify)
    )
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
