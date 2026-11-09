"""CppComponent bridge: declarative native-component construction from the validated config,
delegation, and factory integration. The native class here is a pure-Python stand-in so the core
test suite needs no compiler — the compiled pybind11 counterpart lives in examples/bausatz.
"""

from __future__ import annotations

from typing import Literal

import pytest

from bausatz import Component, ComponentFactory, CppComponent, Registry


class FakeKernel:
    """Stands in for a pybind11-exposed C++ class: its ctor takes the config fields (and the
    component's name) as keyword arguments, exactly as a pybind ``py::arg``-bound constructor would."""

    instances = 0

    def __init__(self, scale: float, bias: float, name: str = "") -> None:
        FakeKernel.instances += 1
        self.scale = scale
        self.bias = bias
        self.name = name

    def apply(self, x: float) -> float:
        return self.scale * x + self.bias


# Declarative: name the native class with `component=`; no _create_component needed. The config
# fields (scale, bias) marshal straight into FakeKernel's keyword arguments.
class Affine(CppComponent, component=FakeKernel):
    class Config(Component.Config):
        class_name: Literal["affine-stub"] = "affine-stub"
        scale: float = 1.0
        bias: float = 0.0

    config: Affine.Config

    def forward(self, x: float) -> float:
        return self.component.apply(x)


@pytest.fixture
def factory() -> ComponentFactory:
    f = ComponentFactory(Registry())
    f.register("affine-stub", Affine)
    return f


def test_builds_from_spec_and_delegates_to_component(factory):
    comp = factory.create({"class_name": "affine-stub", "scale": 3.0, "bias": 1.0})
    assert isinstance(comp, Affine)
    assert comp.forward(2.0) == 7.0  # 3*2 + 1, computed by the native component
    assert isinstance(comp.component, FakeKernel)


def test_component_is_built_lazily_once_and_cached():
    before = FakeKernel.instances
    comp = Affine(Affine.Config(scale=2.0, bias=0.0))
    assert FakeKernel.instances == before  # not built at construction
    native = comp.component
    assert FakeKernel.instances == before + 1  # built on first access
    assert comp.component is native  # cached, not rebuilt


def test_declarative_marshals_config_fields_and_name(factory):
    # class_name is framework-only (not passed); the config fields AND the name reach the ctor.
    comp = factory.create({"class_name": "affine-stub", "name": "enc", "scale": 4.0, "bias": 2.0})
    assert comp.component.scale == 4.0 and comp.component.bias == 2.0
    assert comp.component.name == "enc"          # name passed to the constructor
    # An unnamed component gets "" (config name None -> ""), not None.
    assert factory.create({"class_name": "affine-stub"}).component.name == ""


def test_config_still_validates_and_round_trips(factory):
    # The pydantic Config keeps doing its job even though compute is "native".
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        factory.create({"class_name": "affine-stub", "scale": "not-a-number"})

    comp = factory.create({"class_name": "affine-stub", "scale": 5.0})
    assert comp.to_dict() == {"name": None, "class_name": "affine-stub", "scale": 5.0, "bias": 0.0}
    assert factory.create(comp.to_dict()).forward(1.0) == comp.forward(1.0)


def test_component_fields_hook_can_remap(factory):
    # A component whose config field names differ from the native ctor args overrides _component_fields.
    class Renamed(CppComponent, component=FakeKernel):
        class Config(Component.Config):
            class_name: Literal["renamed-stub"] = "renamed-stub"
            gain: float = 1.0

        def _component_fields(self):
            return {"scale": self.config.gain, "bias": 0.0}

    factory.register("renamed-stub", Renamed)
    assert factory.create({"class_name": "renamed-stub", "gain": 6.0}).component.scale == 6.0


def test_name_excluded_from_dump_is_not_passed(factory):
    # A native class without a `name` parameter: excluding the field from dumps keeps it out of
    # the constructor kwargs (it used to be re-inserted as "").
    from pydantic import Field

    class Nameless:
        def __init__(self, gain: float) -> None:
            self.gain = gain

    class Gain(CppComponent, component=Nameless):
        class Config(Component.Config):
            class_name: Literal["gain-stub"] = "gain-stub"
            name: str | None = Field(None, exclude=True)
            gain: float = 1.0

    factory.register("gain-stub", Gain)
    assert factory.create({"class_name": "gain-stub", "name": "g", "gain": 3.0}).component.gain == 3.0


class FakeChain:
    """Stands in for a native composite (like the example's C++ ``Sequential``)."""

    def __init__(self, stages: list[FakeKernel], name: str = "") -> None:
        self.stages = list(stages)
        self.name = name

    def apply(self, x: float) -> float:
        for stage in self.stages:
            x = stage.apply(x)
        return x


class Chain(CppComponent):
    """A container CppComponent: its native object wraps its children's native objects."""

    class Config(Component.Config):
        class_name: Literal["chain-stub"] = "chain-stub"
        stages: list[Component.Config]

    config: Chain.Config

    def __init__(self, config: Chain.Config) -> None:
        super().__init__(config)
        self.stages: list[Affine] = []
        self.builds = 0

    def _build_children(self, factory: ComponentFactory) -> None:
        self.builds += 1
        self.stages = [factory.build(cfg) for cfg in self.config.stages]

    def _create_component(self) -> FakeChain:
        return FakeChain([stage.component for stage in self.stages], name=self.config.name or "")


def test_directly_constructed_container_builds_its_children_first():
    # Constructed directly (no factory), first access to `component` builds the children before
    # creating the native object, instead of caching a native chain with no stages.
    chain = Chain(Chain.Config(stages=[{"class_name": "affine-stub", "scale": 2.0},
                                       {"class_name": "affine-stub", "bias": 5.0}]))
    assert len(chain.component.stages) == 2
    assert chain.component.apply(3.0) == 11.0  # 2*3, then +5
    assert chain.builds == 1


def test_factory_built_container_does_not_rebuild_its_children(factory):
    factory.register("chain-stub", Chain)
    chain = factory.create({"class_name": "chain-stub", "stages": [{"class_name": "affine-stub"}]})
    assert chain.builds == 1  # built by the factory
    assert len(chain.component.stages) == 1
    assert chain.builds == 1  # `component` found them built: ensure_built() was a no-op


def test_concrete_component_without_native_class_is_rejected_at_definition():
    # A concrete CppComponent that neither declares `component=` nor overrides _create_component
    # fails when the class is defined — not later, at first use.
    with pytest.raises(TypeError, match="no native component to build"):

        class Broken(CppComponent):
            class Config(Component.Config):
                class_name: Literal["broken-stub"] = "broken-stub"
