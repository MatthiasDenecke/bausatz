"""A C++-backed bausatz component.

Wiring, end to end:

    spec dict ─▶ ComponentFactory ─▶ Affine(CppComponent)      [Python: pydantic Config]
                                          │  _make_engine(self.config.scale, self.config.bias)
                                          ▼
                                     AffineKernel(...)          [C++: pybind11 extension _affine]

``Affine`` is an ordinary component — it self-registers as ``"affine"`` and the factory builds it
from a spec like any pure-Python component. The only difference is that its ``forward`` delegates to
the compiled ``AffineKernel``.
"""

from __future__ import annotations

from typing import Literal

from bausatz import Component, CppComponent

# The pybind11 extension module built from ../src/affine.cpp. Re-exported so callers can also use
# the raw kernel directly if they want.
from ._affine import AffineKernel

__all__ = ["Affine", "AffineKernel"]


class Affine(CppComponent):
    """y = scale * x + bias, computed in C++."""

    class Config(Component.Config):
        class_name: Literal["affine"] = "affine"
        scale: float = 1.0
        bias: float = 0.0

    config: Affine.Config

    def _make_engine(self) -> AffineKernel:
        # self.config is already validated by pydantic; hand the plain scalars to the C++ ctor.
        return AffineKernel(self.config.scale, self.config.bias)

    def forward(self, x: float) -> float:
        return self.engine.apply(x)

    def forward_many(self, xs: list[float]) -> list[float]:
        return self.engine.apply_many(xs)
