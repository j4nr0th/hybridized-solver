"""Session-wide guards for the test suite.

The extension is installed as a scikit-build-core *editable* package with
``editable.rebuild = false``, because rebuild-on-import is not usable here (see
the note in ``pyproject.toml``). The cost of a fast import is that an edit to
the C sources does not reach the interpreter, and the suite then passes against
whatever was built last. That is the worst possible failure mode for a change
that is mostly C, so it is checked here rather than left to be noticed.
"""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path

import hybsol._mod
import pytest

# Everything compiled into the extension. ``cutl`` and ``cpyutl`` are vendored,
# but their headers sit on the extension's include path, so a change to either
# is as good a reason to rebuild as a change to hybsol itself.
_SOURCE_DIRS = ("src", "include", "cutl", "cpyutl")
_SOURCE_SUFFIXES = {".c", ".h", ".inc", ".pyi"}


def _stamp(path: Path) -> str:
    when = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return when.isoformat(timespec="seconds")


def _newest_source(repo: Path) -> tuple[float, Path] | None:
    newest: tuple[float, Path] | None = None
    for directory in _SOURCE_DIRS:
        root = repo / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if newest is None or mtime > newest[0]:
                newest = (mtime, path)
    return newest


@pytest.fixture(scope="session", autouse=True)
def extension_is_current(repo_root: Path):
    """Fail the run if an installed extension is older than its sources."""
    newest = _newest_source(repo_root)
    if newest is None:
        pytest.skip("no C sources found to compare against")
    newest_mtime, newest_path = newest

    _check_stale(Path(hybsol._mod.__file__), repo_root, newest_mtime, newest_path)

    # The backend's extension is a separate build of separate sources, and an
    # installation without the backend has none: only check what is there.
    backend = importlib.util.find_spec("hybsol._mod_opencl")
    if backend is not None and backend.origin is not None:
        _check_stale(Path(backend.origin), repo_root, newest_mtime, newest_path)


def _check_stale(
    extension: Path, repo_root: Path, newest_mtime: float, newest_path: Path
) -> None:
    """Fail the run when ``extension`` predates the newest source."""
    if extension.stat().st_mtime >= newest_mtime:
        return
    pytest.fail(
        f"the installed extension is stale.\n"
        f"  extension:     {extension} ({_stamp(extension)})\n"
        f"  newest source: {newest_path.relative_to(repo_root)}"
        f" ({_stamp(newest_path)})\n"
        f"The tests would run against the previous build. Rebuild with:\n"
        f"  uv sync --reinstall-package hybsol",
        pytrace=False,
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the checkout root, for tests that reach outside ``tests/``."""
    return Path(__file__).resolve().parent.parent
