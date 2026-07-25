"""Config-driven component framework: build classes from dict/JSON specs by a ``class_name`` tag.

A ``Component`` declares a typed nested ``Config`` and self-registers on import; ``ComponentFactory``
turns a spec into an instance; ``Registry`` is the tag -> class table. See ``component.py`` for the
registration mechanics.

The library also ships C++ headers for the native-component bridge (``bausatz::Component``); a
downstream pybind11 extension locates them with ``bausatz.get_include()``.
"""

import os

from bausatz.component import Component
from bausatz.component_cpp import CppComponent
from bausatz.component_factory import ComponentFactory
from bausatz.registry import Registry


def get_include() -> str:
    """Directory holding the bausatz C++ headers, for a downstream pybind11 extension::

        import bausatz
        Pybind11Extension(..., include_dirs=[bausatz.get_include()])  # #include <bausatz/component.hpp>

    In the repo the headers live at the top-level ``inc/``; an installed wheel bundles them into the
    package (see ``setup.py``). This returns whichever exists.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    packaged = os.path.join(here, "inc")  # installed wheel: headers copied into the package
    if os.path.isdir(packaged):
        return packaged
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir, "inc"))  # repo: top-level inc/


__all__ = ["Component", "ComponentFactory", "CppComponent", "Registry", "get_include"]
