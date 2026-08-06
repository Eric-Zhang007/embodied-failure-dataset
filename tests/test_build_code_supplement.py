import importlib.util
from pathlib import Path
from zipfile import ZipFile


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_code_supplement.py"
SPEC = importlib.util.spec_from_file_location("build_code_supplement", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(root: Path, relative: str, content: str = "pass\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_builds_anonymous_allowlisted_archive(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    _write(source, "src/__init__.py", "")
    _write(source, "src/runner.py")
    _write(source, "scripts/run_pipeline.py")
    _write(source, "scripts/analyze_collection_snapshot.py")
    _write(source, "pyproject.toml", "[project]\nname = 'ef-bench'\n")
    _write(source, "uv.lock", "version = 1\n")

    _write(source, "tests/test_runner.py")
    _write(source, "output_run/episode.json", "{}\n")
    _write(source, "work/notes.md", "private\n")
    _write(source, "paper/main.tex", "draft\n")
    _write(source, "docs/plan.md", "plan\n")
    _write(source, "config.toml", "api_key = 'secret-value'\n")
    _write(source, "README.md", "github.com/Eric-Zhang007\n")

    archive = tmp_path / "efbench-code.zip"
    report = MODULE.build_code_supplement(source, archive)

    assert report["file_count"] == 8
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert names == {
            "README.md",
            "config.example.toml",
            "pyproject.toml",
            "uv.lock",
            "src/__init__.py",
            "src/runner.py",
            "scripts/run_pipeline.py",
            "scripts/analyze_collection_snapshot.py",
        }
        combined = b"\n".join(bundle.read(name) for name in names)

    assert b"secret-value" not in combined
    assert b"Eric-Zhang007" not in combined
    assert MODULE.verify_code_supplement(archive)["file_count"] == 8


def test_verifier_rejects_forbidden_paths(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("tests/test_bad.py", "pass\n")

    try:
        MODULE.verify_code_supplement(archive)
    except ValueError as error:
        assert "forbidden" in str(error).lower()
    else:
        raise AssertionError("forbidden test path was accepted")
