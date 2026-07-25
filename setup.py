"""Project metadata lives in ``pyproject.toml``. This file exists only to bundle the top-level C++
headers (``inc/``) into the built package, so ``bausatz.get_include()`` resolves inside an installed
wheel — where there is no repo-level ``inc/`` to fall back to.

The headers stay at the repo top level (``inc/bausatz/*.hpp``); at build time they are copied into
the package as ``bausatz/inc/bausatz/*.hpp``. Editable installs use the source tree directly, so the
copy doesn't run and ``get_include()`` falls back to the top-level ``inc/``.
"""

import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class build_py_with_headers(build_py):
    def run(self) -> None:
        super().run()
        repo_root = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(repo_root, "inc")
        if os.path.isdir(src):
            dst = os.path.join(self.build_lib, "bausatz", "inc")
            shutil.copytree(src, dst, dirs_exist_ok=True)


setup(cmdclass={"build_py": build_py_with_headers})
