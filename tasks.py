#!/usr/bin/env python
"""indic-er task runner.

Cross-platform stand-in for `make`, so the project has one entry point on
machines without GNU make (Windows in particular).

    py tasks.py <target> [args...]

Targets: setup | data | test | lint | fmt | eval | demo | api | clean | help

`Makefile` and `make.bat` both delegate here, so `make demo` and
`py tasks.py demo` do exactly the same thing.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
DEMO_HOST = os.environ.get("INDIC_ER_HOST", "127.0.0.1")
DEMO_PORT = os.environ.get("INDIC_ER_PORT", "8141")


def sh(*args: str, check: bool = True, **kwargs) -> int:
    """Run a command, echoing it first."""
    printable = " ".join(str(a) for a in args)
    print(f"\n$ {printable}", flush=True)
    proc = subprocess.run([str(a) for a in args], cwd=ROOT, **kwargs)
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc.returncode


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}", flush=True)


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #


def t_setup(_: list[str]) -> None:
    """Install Python dependencies."""
    banner("setup — installing dependencies")
    sh(PY, "-m", "pip", "install", "-r", "requirements.txt")


def t_data(argv: list[str]) -> None:
    """Generate the seeded synthetic dataset."""
    banner("data — generating synthetic records")
    sh(PY, "data/generate.py", *argv)


def t_lint(_: list[str]) -> None:
    """Lint with ruff."""
    banner("lint — ruff check")
    sh(PY, "-m", "ruff", "check", "app", "tests", "data", "tasks.py")
    sh(PY, "-m", "ruff", "format", "--check", "app", "tests", "data", "tasks.py", check=False)


def t_fmt(_: list[str]) -> None:
    """Auto-format with ruff."""
    banner("fmt — ruff format")
    sh(PY, "-m", "ruff", "format", "app", "tests", "data", "tasks.py")
    sh(PY, "-m", "ruff", "check", "--fix", "app", "tests", "data", "tasks.py", check=False)


def t_test(argv: list[str]) -> None:
    """Run the test suite with coverage on app/core."""
    banner("test — pytest + coverage on app/core")
    ensure_data()
    sh(
        PY,
        "-m",
        "pytest",
        "--cov=app/core",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        *argv,
    )


def t_eval(argv: list[str]) -> None:
    """Run the evaluation and write docs/eval-report.md."""
    banner("eval — baseline vs pipeline, sweep, ablations")
    ensure_data()
    sh(PY, "-m", "app.evaluation.run_eval", *argv)


def t_api(argv: list[str]) -> None:
    """Run the API (and web app) without opening a browser."""
    ensure_data()
    serve(open_browser=False, extra=argv)


def t_demo(argv: list[str]) -> None:
    """Seed everything and boot API + web, then open a browser."""
    banner("demo — seeding and booting indic-er")
    ensure_data()
    seed_store()
    serve(open_browser="--no-open" not in argv, extra=[a for a in argv if a != "--no-open"])


def t_clean(_: list[str]) -> None:
    """Remove generated artefacts and caches."""
    banner("clean")
    targets = [
        ROOT / ".pytest_cache",
        ROOT / ".ruff_cache",
        ROOT / "htmlcov",
        ROOT / ".coverage",
        ROOT / "data" / "records.csv",
        ROOT / "data" / "ground_truth.csv",
        ROOT / "data" / "records.json",
        ROOT / "data" / "indic_er.db",
    ]
    for pyc in ROOT.rglob("__pycache__"):
        targets.append(pyc)
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"  removed dir  {target.relative_to(ROOT)}")
        elif target.exists():
            target.unlink()
            print(f"  removed file {target.relative_to(ROOT)}")
    print("clean done")


def t_help(_: list[str]) -> None:
    print(__doc__)
    width = max(len(name) for name in TARGETS)
    print("Targets:")
    for name, fn in TARGETS.items():
        print(f"  {name.ljust(width)}  {(fn.__doc__ or '').splitlines()[0]}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def ensure_data() -> None:
    """Generate the dataset if it is not on disk yet."""
    if not (ROOT / "data" / "records.csv").exists():
        print("[tasks] data/records.csv missing — generating it first")
        sh(PY, "data/generate.py")


def seed_store() -> None:
    """Resolve the seeded dataset into the graph store and review queue."""
    print("[tasks] seeding graph store + review queue")
    sh(PY, "-m", "app.api.seed")


def serve(*, open_browser: bool, extra: list[str]) -> None:
    url = f"http://{DEMO_HOST}:{DEMO_PORT}"
    print(f"\n  indic-er is starting on {url}")
    print(f"    playground   {url}/")
    print(f"    api docs     {url}/docs")
    print(f"    health       {url}/health")
    print("\n  Ctrl-C to stop.\n", flush=True)
    if open_browser:
        with contextlib.suppress(Exception):  # headless or no default browser
            webbrowser.open(url)
    sh(
        PY,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        DEMO_HOST,
        "--port",
        DEMO_PORT,
        *extra,
        check=False,
    )


TARGETS = {
    "setup": t_setup,
    "data": t_data,
    "lint": t_lint,
    "fmt": t_fmt,
    "test": t_test,
    "eval": t_eval,
    "api": t_api,
    "demo": t_demo,
    "clean": t_clean,
    "help": t_help,
}


def main() -> None:
    argv = sys.argv[1:]
    target = argv[0] if argv else "help"
    if target in {"-h", "--help"}:
        target = "help"
    fn = TARGETS.get(target)
    if fn is None:
        print(f"unknown target: {target!r}\n")
        t_help([])
        sys.exit(2)
    fn(argv[1:])


if __name__ == "__main__":
    main()
