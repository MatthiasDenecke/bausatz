"""Bridge for components whose behavior is implemented natively — typically a C++ class exposed
with pybind11, though any Python-callable object works as the "engine".

The division of labor is the whole point:

  * The pydantic ``Config`` keeps everything that makes a component config-driven — field
    validation, the ``class_name`` tag, factory registration, ``to_dict`` round-trip — all in
    Python, where pydantic lives. The C++ side never parses or validates a spec.
  * The native **engine** owns the compute. It is constructed from the *already-validated* config,
    so crossing the language boundary is just handing plain numbers/strings/lists (pybind11's type
    casters convert them) to a constructor.

A ``CppComponent`` is an ordinary ``Component``: it self-registers by its tag and the
``ComponentFactory`` builds it from a spec exactly like any other. It simply delegates its methods
to the engine. Subclasses implement ``_make_engine`` to construct the native object from
``self.config``:

    class Affine(CppComponent):
        class Config(Component.Config):
            class_name: Literal["affine"] = "affine"
            scale: float = 1.0
            bias: float = 0.0
        config: Affine.Config

        def _make_engine(self):
            from bausatz_affine import AffineKernel      # the pybind11 extension type
            return AffineKernel(self.config.scale, self.config.bias)

        def forward(self, x: float) -> float:
            return self.engine.apply(x)                  # straight into C++
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bausatz.component import Component

_UNSET: Any = object()  # a private sentinel, distinct from a legitimate ``None`` engine


class CppComponent(Component, ABC):
    """A ``Component`` whose behavior is provided by a native engine object.

    Abstract: a subclass must implement ``_make_engine``, so forgetting to fails at construction —
    not later, at first ``engine`` access. (Unlike ``Component``, whose only "abstract" contract is
    a pinned ``class_name``, this base has a genuine abstract method, so ``ABC`` earns its place.)

    The engine is built once, lazily, from the validated config and cached — so a ``CppComponent``
    is usable whether it was assembled by the factory or constructed directly, with no separate
    build step to remember. To fail fast at *build* time instead (surfacing a bad engine while the
    tree is being assembled), override ``_build_children`` to touch ``self.engine``:

        def _build_children(self, factory):
            _ = self.engine  # construct eagerly during ensure_built()
    """

    def __init__(self, config: Component.Config) -> None:
        super().__init__(config)
        self._engine: Any = _UNSET  # built on first ``engine`` access

    @abstractmethod
    def _make_engine(self) -> Any:
        """Construct and return the native engine from ``self.config``. Implement in subclasses."""

    @property
    def engine(self) -> Any:
        """The native engine, built from the config on first access and cached thereafter."""
        if self._engine is _UNSET:
            self._engine = self._make_engine()
        return self._engine
