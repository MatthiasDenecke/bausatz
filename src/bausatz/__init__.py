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
    """
    return os.path.join(os.path.dirname(__file__), "include")


__all__ = ["Component", "ComponentFactory", "CppComponent", "Registry", "get_include"]
