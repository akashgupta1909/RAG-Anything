"""Every package must import on its own, in a fresh interpreter.

Within one test session, import order can hide circular imports (a module
that only works because another one happened to be imported first).
"""

import pkgutil
import subprocess
import sys

import pytest

import rag

PACKAGES = sorted(
    name for _, name, is_pkg in pkgutil.walk_packages(rag.__path__, "rag.") if is_pkg
)


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports_in_isolation(package: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {package}"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1:]
