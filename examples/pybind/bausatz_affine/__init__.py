"""C++-backed bausatz components.

Two shapes, both driven from Python as ordinary config-driven components:

    spec dict ─▶ ComponentFactory ─▶ Affine(CppComponent)          [Python: pydantic Config]
                                          │  _make_engine: AffineConfig(scale, bias)  ← config struct
                                          ▼
                                     AffineKernel : bausatz::Component   [C++: extension _affine]

    spec dict ─▶ ComponentFactory ─▶ NativeSequential(CppComponent)
                                          │  _build_children → child Affines
                                          │  _make_engine: Sequential([child.engine, ...])
                                          ▼
                                     Sequential : bausatz::Component     ← runs the chain in C++

``Affine`` shows config-struct passing (the pydantic config becomes a typed C++ ``AffineConfig``).
``NativeSequential`` shows the native-composite shape: its engine is a C++ ``Sequential`` that holds
the child kernels by ``bausatz::Component`` base pointer and runs the whole chain in C++, with no
per-stage bounce back to Python. Its children must themselves be C++-backed (their ``engine`` is a
``bausatz::Component``).
"""

from __future__ import annotations

from typing import Literal

from bausatz import Component, ComponentFactory, CppComponent

# The pybind11 extension built from ../src/affine.cpp:
#   Component     - the C++ abstract base (bausatz::Component), re-exported for isinstance checks
#   AffineConfig  - the mirrored C++ config struct
#   AffineKernel  - a leaf engine
#   Sequential    - a native composite engine
from ._affine import AffineConfig, AffineKernel
from ._affine import Component as NativeComponent
from ._affine import Sequential as _SequentialEngine

__all__ = ["Affine", "AffineConfig", "AffineKernel", "NativeComponent", "NativeSequential"]


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
        return AffineKernel(AffineConfig(scale=self.config.scale, bias=self.config.bias))

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
        # Hand each child's native engine (a bausatz::Component) to the C++ Sequential, which owns
        # them via shared_ptr and runs the chain natively.
        return _SequentialEngine([stage.engine for stage in self.stages])

    def forward(self, x: float) -> float:
        return self.engine.forward(x)
