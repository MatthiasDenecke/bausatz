"""Config-driven component framework: build classes from dict/JSON specs by a ``class_name`` tag.

A ``Component`` declares a typed nested ``Config`` and self-registers on import; ``ComponentFactory``
turns a spec into an instance; ``Registry`` is the tag -> class table. See ``component.py`` for the
registration mechanics.
"""

from bausatz.registry import Registry
from bausatz.component import Component
from bausatz.component_factory import ComponentFactory

__all__ = ["Component", "ComponentFactory", "Registry"]
