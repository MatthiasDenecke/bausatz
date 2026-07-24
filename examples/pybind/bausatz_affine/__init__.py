"""C++-backed bausatz components.

Two shapes, both driven from Python as ordinary config-driven components:

    spec dict ─▶ ComponentFactory ─▶ Affine(CppComponent)          [Python: pydantic Config]
                                          │  _make_engine: AffineConfig(scale, bias)  ← config struct
                                          ▼
                                     AffineKernel : Transform : bausatz::Component   [C++]

    spec dict ─▶ ComponentFactory ─▶ NativeSequential(CppComponent)
                                          │  _build_children → child Affines
                                          │  _make_engine: Sequential([child.engine, ...])
                                          ▼
                                     Sequential : Transform : bausatz::Component  ← chain runs in C++

``bausatz::Component`` (the library's C++ base, via ``bausatz.get_include()``) carries no compute
method; the example defines its own compute interface ``Transform`` (that's where ``forward`` lives).
``Affine`` shows config-struct passing; ``NativeSequential``'s engine is a C++ ``Sequential`` that
holds the child transforms by base pointer and runs the whole chain in C++. Its children must
themselves be C++-backed (their ``engine`` is a ``Transform``).
"""

from __future__ import annotations

from typing import Literal

from bausatz import Component, ComponentFactory, CppComponent

# The pybind11 extension built from ../src/affine.cpp:
#   Component     - the library C++ base (bausatz::Component), re-exported for isinstance checks
#   Transform     - the example's compute interface (defines forward), deriving from Component
#   AffineConfig  - the mirrored C++ config struct
#   AffineKernel  - a leaf transform
#   Sequential    - a native composite transform
from ._affine import AffineConfig, AffineKernel, Transform
from ._affine import Component as NativeComponent
from ._affine import Sequential as _SequentialEngine

__all__ = [
    "Affine",
    "AffineConfig",
    "AffineKernel",
    "NativeComponent",
    "NativeSequential",
    "Transform",
]


class Affine(CppComponent):
    """y = scale * x + bias, computed in C++."""

    class Config(Component.Config):
        class_name: Literal["affine"] = "affine"
        scale: float = 1.0
        bias: float = 0.0

    config: Affine.Config

    def _make_engine(self) -> AffineKernel:
        # Config-struct passing: build the typed C++ AffineConfig from the validated pydantic config,
        # then hand that one object to the kernel (rather than a positional arg list).
        kernel = AffineKernel(AffineConfig(scale=self.config.scale, bias=self.config.bias))
        if self.config.name:
            kernel.name = self.config.name  # carry the config name onto the C++ base (plumbing)
        return kernel

    def forward(self, x: float) -> float:
        return self.engine.forward(x)

    def forward_many(self, xs: list[float]) -> list[float]:
        return self.engine.forward_many(xs)


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

    def _make_engine(self) -> _SequentialEngine:
        # Hand each child's native engine (a Transform) to the C++ Sequential, which owns them via
        # shared_ptr and runs the chain natively.
        return _SequentialEngine([stage.engine for stage in self.stages])

    def forward(self, x: float) -> float:
        return self.engine.forward(x)
