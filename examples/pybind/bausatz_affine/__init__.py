"""C++-backed bausatz components.

Two shapes, both driven from Python as ordinary config-driven components:

    spec dict ─▶ ComponentFactory ─▶ Affine(CppComponent, component=AffineKernel)   [Python]
                                          │  default _create_component: AffineKernel(**config fields)
                                          ▼
                                     AffineKernel : Transform : bausatz::Component   [C++]

    spec dict ─▶ ComponentFactory ─▶ NativeSequential(CppComponent)
                                          │  _build_children → child Affines
                                          │  _create_component: Sequential([child.component, ...])
                                          ▼
                                     Sequential : Transform : bausatz::Component  ← chain runs in C++

``bausatz::Component`` (the library's C++ base, via ``bausatz.get_include()``) carries no compute
method; the example defines its own compute interface ``Transform`` (that's where ``forward`` lives).
``Affine`` needs no construction code at all — it just names its native class with ``component=`` and
the framework marshals the validated config fields in as keyword arguments. ``NativeSequential``
overrides ``_create_component`` because it must build child components first, then wrap their native
``component``s in a C++ ``Sequential`` that runs the whole chain in C++. Its children must themselves
be C++-backed (their ``component`` is a ``Transform``).
"""

from __future__ import annotations

from typing import Literal

from bausatz import Component, ComponentFactory, CppComponent

# The pybind11 extension built from ../src/affine.cpp:
#   Component     - the library C++ base (bausatz::Component), re-exported for isinstance checks
#   Transform     - the example's compute interface (defines forward), deriving from Component
#   AffineKernel  - a leaf transform whose ctor takes (scale, bias) keyword args
#   Sequential    - a native composite transform
from ._affine import AffineKernel, Transform
from ._affine import Component as NativeComponent
from ._affine import Sequential as _SequentialComponent

__all__ = ["Affine", "AffineKernel", "NativeComponent", "NativeSequential", "Transform"]


class Affine(CppComponent, component=AffineKernel):
    """y = scale * x + bias, computed in C++.

    Declarative: naming ``component=AffineKernel`` is all it takes — the framework builds
    ``AffineKernel(scale=..., bias=...)`` from the validated config, and carries ``config.name`` onto
    the native component. No ``_create_component``.
    """

    class Config(Component.Config):
        class_name: Literal["affine"] = "affine"
        scale: float = 1.0
        bias: float = 0.0

    config: Affine.Config

    def forward(self, x: float) -> float:
        return self.component.forward(x)

    def forward_many(self, xs: list[float]) -> list[float]:
        return self.component.forward_many(xs)


class NativeSequential(CppComponent):
    """Runs a chain of C++-backed stages entirely in C++ (one native call per ``forward``)."""

    class Config(Component.Config):
        class_name: Literal["native-sequential"] = "native-sequential"
        stages: list[Component.Config]      # typed children, resolved eagerly by the framework

    config: NativeSequential.Config

    def __init__(self, config: NativeSequential.Config) -> None:
        super().__init__(config)
        self.stages: list[Component] = []

    def _build_children(self, factory: ComponentFactory) -> None:
        # Build the child components against the same factory the framework is using.
        self.stages = [factory.build(cfg) for cfg in self.config.stages]

    def _create_component(self) -> _SequentialComponent:
        # Custom construction: hand each child's native component (a Transform) to the C++ Sequential,
        # which owns them via shared_ptr and runs the chain natively.
        return _SequentialComponent(
            [stage.component for stage in self.stages], name=self.config.name or ""
        )

    def forward(self, x: float) -> float:
        return self.component.forward(x)
