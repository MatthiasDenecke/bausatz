"""
Component framework: self-registration, tag-dispatched construction,
recursive child building, config validation, and the discriminated-union adapter.

The Linear / Dropout / Sequential classes below are throwaway test doubles that
exercise the three component shapes: a leaf (Linear), a leaf with a constrained
field (Dropout), and a container that builds nested children (Sequential).
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field, ValidationError

from bausatz import ChildConfig, Component, ComponentFactory, Registry
from bausatz.registry import DuplicateTagError, UnknownTagError


# --------------------------------------------------------------------------- #
# Test components (auto-register into the global factory on definition).
# --------------------------------------------------------------------------- #
class Linear(Component):
    class Config(Component.Config):
        class_name: Literal["linear"] = "linear"
        in_features: int
        out_features: int
        bias: bool = True

    config: Linear.Config

    def forward(self, x: float) -> float:
        return x * self.config.out_features + (1.0 if self.config.bias else 0.0)


class Dropout(Component):
    class Config(Component.Config):
        class_name: Literal["dropout"] = "dropout"
        p: float = Field(ge=0.0, le=1.0)

    config: Dropout.Config

    def forward(self, x: float) -> float:
        return x * (1.0 - self.config.p)


class Sequential(Component):
    class Config(Component.Config):
        class_name: Literal["sequential"] = "sequential"
        layers: list[ChildConfig]  # child specs, each resolved to its concrete Config eagerly

    config: Sequential.Config

    def __init__(self, config: Sequential.Config) -> None:
        super().__init__(config)
        self.layers: list[Component] = []

    def _build_children(self, factory: ComponentFactory) -> None:
        self.layers = [factory.build(cfg) for cfg in self.config.layers]

    def forward(self, x: float) -> float:
        for layer in self.layers:
            x = layer.forward(x)
        return x


@pytest.fixture
def factory() -> ComponentFactory:
    """A fresh, isolated factory with the three test components registered.

    Isolated from the process-global singleton so tag/adapter assertions stay
    deterministic regardless of what else the test session has registered.
    """
    f = ComponentFactory(Registry())
    f.register("linear", Linear)
    f.register("dropout", Dropout)
    f.register("sequential", Sequential)
    return f


# --------------------------------------------------------------------------- #
# Self-registration via __init_subclass__.
# --------------------------------------------------------------------------- #
def test_components_self_register_in_global_factory():
    reg = ComponentFactory.get().registry
    assert reg.get("linear") is Linear
    assert reg.get("dropout") is Dropout
    assert reg.get("sequential") is Sequential


def test_class_name_without_default_is_rejected():
    with pytest.raises(TypeError, match="needs a default"):

        class Bad(Component):
            class Config(Component.Config):
                class_name: Literal["bad"]  # no default -> cannot self-register


def test_component_without_discriminator_is_abstract():
    # No class_name pinned -> treated as an abstract base, silently not registered.
    class Abstract(Component):
        class Config(Component.Config):
            pass

    assert "class_name" not in Abstract.Config.model_fields


# --------------------------------------------------------------------------- #
# Construction: create / create_many, recursive children.
# --------------------------------------------------------------------------- #
def test_create_returns_typed_instance(factory):
    obj = factory.create(
        {"class_name": "linear", "name": "enc", "in_features": 128, "out_features": 64}
    )
    assert isinstance(obj, Linear)
    assert obj.config.name == "enc"
    assert obj.config.in_features == 128
    assert obj.config.bias is True  # default applied


def test_create_many_builds_named_components(factory):
    config = {
        "encoder": {
            "class_name": "linear",
            "name": "enc",
            "in_features": 128,
            "out_features": 64,
        },
        "regularizer": {"class_name": "dropout", "p": 0.5},
        "head": {
            "class_name": "sequential",
            "layers": [
                {"class_name": "linear", "in_features": 64, "out_features": 8, "bias": False},
                {"class_name": "dropout", "p": 0.1},
            ],
        },
    }
    built = factory.create_many(config)

    assert set(built) == {"encoder", "regularizer", "head"}
    assert isinstance(built["encoder"], Linear)
    assert isinstance(built["regularizer"], Dropout)
    assert isinstance(built["head"], Sequential)


def test_container_builds_nested_children(factory):
    head = factory.create(
        {
            "class_name": "sequential",
            "layers": [
                {"class_name": "linear", "in_features": 64, "out_features": 8, "bias": False},
                {"class_name": "dropout", "p": 0.1},
            ],
        }
    )
    assert [type(layer).__name__ for layer in head.layers] == ["Linear", "Dropout"]


# --------------------------------------------------------------------------- #
# Behavior: forward passes.
# --------------------------------------------------------------------------- #
def test_linear_forward(factory):
    enc = factory.create({"class_name": "linear", "in_features": 128, "out_features": 64})
    assert enc.forward(2.0) == 2.0 * 64 + 1.0  # bias on


def test_dropout_forward(factory):
    reg = factory.create({"class_name": "dropout", "p": 0.5})
    assert reg.forward(2.0) == 1.0


def test_sequential_forward_chains_children(factory):
    head = factory.create(
        {
            "class_name": "sequential",
            "layers": [
                {"class_name": "linear", "in_features": 64, "out_features": 8, "bias": False},
                {"class_name": "dropout", "p": 0.1},
            ],
        }
    )
    # linear (8x, no bias) -> 16.0 ; dropout (0.9x) -> 14.4
    assert head.forward(2.0) == pytest.approx(14.4)


# --------------------------------------------------------------------------- #
# Serialization round-trip.
# --------------------------------------------------------------------------- #
def test_to_dict_round_trips_through_create(factory):
    enc = factory.create(
        {"class_name": "linear", "name": "enc", "in_features": 128, "out_features": 64}
    )
    clone = factory.create(enc.to_dict())
    assert isinstance(clone, Linear)
    assert clone.config == enc.config


def test_to_dict_preserves_discriminator(factory):
    enc = factory.create({"class_name": "linear", "in_features": 4, "out_features": 2})
    assert enc.to_dict()["class_name"] == "linear"


def test_to_dict_round_trips_nested_children(factory):
    # SerializeAsAny must emit each child's concrete fields (bias, p), not just the base config.
    spec = {
        "class_name": "sequential",
        "layers": [
            {"class_name": "linear", "in_features": 64, "out_features": 8, "bias": False},
            {"class_name": "dropout", "p": 0.1},
        ],
    }
    head = factory.create(spec)
    clone = factory.create(head.to_dict())
    assert isinstance(clone, Sequential)
    assert [type(layer).__name__ for layer in clone.layers] == ["Linear", "Dropout"]
    assert clone.config == head.config
    dumped = head.to_dict()
    assert dumped["layers"][0]["bias"] is False and dumped["layers"][1]["p"] == 0.1


# --------------------------------------------------------------------------- #
# Validation errors.
# --------------------------------------------------------------------------- #
def test_create_requires_mapping_with_class_name(factory):
    with pytest.raises(ValueError, match="class_name"):
        factory.create({"in_features": 4, "out_features": 2})
    with pytest.raises(ValueError, match="class_name"):
        factory.create("linear")  # not a mapping


def test_create_unknown_tag_raises(factory):
    with pytest.raises(UnknownTagError):
        factory.create({"class_name": "does-not-exist"})


def test_config_forbids_unknown_keys(factory):
    with pytest.raises(ValidationError):
        factory.create(
            {"class_name": "linear", "in_features": 4, "out_features": 2, "bogus": 1}
        )


def test_config_enforces_field_constraints(factory):
    with pytest.raises(ValidationError):
        factory.create({"class_name": "dropout", "p": 1.5})  # p must be <= 1.0


def test_config_requires_mandatory_fields(factory):
    with pytest.raises(ValidationError):
        factory.create({"class_name": "linear", "out_features": 2})  # missing in_features


def test_validate_returns_config_without_constructing(factory):
    cfg = factory.validate({"class_name": "linear", "in_features": 4, "out_features": 2})
    assert isinstance(cfg, Linear.Config)
    assert cfg.in_features == 4 and cfg.out_features == 2


# --------------------------------------------------------------------------- #
# Typed children: nested specs validate eagerly into concrete Configs.
# --------------------------------------------------------------------------- #
def test_children_validate_to_concrete_configs(factory):
    cfg = factory.validate(
        {
            "class_name": "sequential",
            "layers": [
                {"class_name": "linear", "in_features": 4, "out_features": 2},
                {"class_name": "dropout", "p": 0.3},
            ],
        }
    )
    assert isinstance(cfg, Sequential.Config)
    assert isinstance(cfg.layers[0], Linear.Config)
    assert isinstance(cfg.layers[1], Dropout.Config)
    assert cfg.layers[0].out_features == 2


def test_validate_rejects_malformed_child_eagerly(factory):
    # A bad grandchild is caught at validate() time, not deferred to build time.
    with pytest.raises(ValidationError):
        factory.validate(
            {"class_name": "sequential", "layers": [{"class_name": "linear", "in_features": 4}]}
        )  # missing out_features
    with pytest.raises(ValidationError):
        factory.validate(
            {"class_name": "sequential", "layers": [{"class_name": "dropout", "p": 5.0}]}
        )  # p must be <= 1.0
    with pytest.raises(ValidationError):
        factory.validate(
            {"class_name": "sequential", "layers": [{"class_name": "linear",
                                                     "in_features": 4, "out_features": 2,
                                                     "bogus": 1}]}
        )  # unknown key in child


def test_build_as_enforces_expected_type(factory):
    cfg = factory.validate({"class_name": "linear", "in_features": 4, "out_features": 2})
    assert isinstance(factory.build_as(cfg, Linear), Linear)  # ok
    with pytest.raises(TypeError, match="not a Dropout"):
        factory.build_as(cfg, Dropout)  # wrong subtype


def test_create_as_enforces_expected_type(factory):
    obj = factory.create_as({"class_name": "dropout", "p": 0.2}, Dropout)
    assert isinstance(obj, Dropout)
    with pytest.raises(TypeError, match="not a Linear"):
        factory.create_as({"class_name": "dropout", "p": 0.2}, Linear)


def test_child_spec_without_class_name_is_rejected(factory):
    with pytest.raises(ValidationError):
        factory.validate({"class_name": "sequential", "layers": [{"in_features": 4}]})


def test_already_resolved_child_config_passes_through(factory):
    # Constructing a container Config with an already-built child Config (not a dict) is accepted
    # as-is and still builds.
    cfg = Sequential.Config(layers=[Linear.Config(in_features=4, out_features=2)])
    assert isinstance(cfg.layers[0], Linear.Config)
    head = factory.build(cfg)
    assert isinstance(head.layers[0], Linear)


def test_build_rejects_config_without_discriminator(factory):
    with pytest.raises(ValueError, match="not buildable"):
        factory.build(Component.Config())  # base config, no class_name


def test_unknown_child_tag_is_rejected(factory):
    with pytest.raises(UnknownTagError):
        factory.validate(
            {"class_name": "sequential", "layers": [{"class_name": "does-not-exist"}]}
        )


# --------------------------------------------------------------------------- #
# Registry semantics.
# --------------------------------------------------------------------------- #
def test_registry_tags_preserve_insertion_order(factory):
    assert factory.registry.tags() == ["linear", "dropout", "sequential"]


def test_registry_rejects_conflicting_tag():
    reg = Registry()
    reg.register("x", Linear)
    with pytest.raises(DuplicateTagError):
        reg.register("x", Dropout)


def test_registry_re_register_same_class_is_idempotent():
    reg = Registry()
    reg.register("x", Linear)
    reg.register("x", Linear)  # no-op, must not raise
    assert reg.get("x") is Linear


def test_registry_unknown_tag_raises():
    reg = Registry()
    with pytest.raises(UnknownTagError):
        reg.get("missing")


# --------------------------------------------------------------------------- #
# Discriminated-union config adapter.
# --------------------------------------------------------------------------- #
def test_config_adapter_routes_by_discriminator(factory):
    adapter = factory.config_adapter()

    parsed = adapter.validate_python({"class_name": "dropout", "p": 0.25})
    assert isinstance(parsed, Dropout.Config)
    assert parsed.p == 0.25

    parsed = adapter.validate_python(
        {"class_name": "linear", "in_features": 2, "out_features": 3}
    )
    assert isinstance(parsed, Linear.Config)


def test_config_adapter_single_config_branch():
    f = ComponentFactory(Registry())
    f.register("dropout", Dropout)
    parsed = f.config_adapter().validate_python({"class_name": "dropout", "p": 0.25})
    assert isinstance(parsed, Dropout.Config)


def test_config_adapter_is_cached_until_registry_grows():
    f = ComponentFactory(Registry())
    f.register("dropout", Dropout)
    first = f.config_adapter()
    assert f.config_adapter() is first  # cached
    f.register("linear", Linear)
    assert f.config_adapter() is not first  # rebuilt after registry grew


def test_config_adapter_rebuilds_after_clear_and_reregister():
    # A version counter (not a size check) catches clear + re-register to the SAME size: the size is
    # unchanged but the registered class is different, so the cached adapter must be discarded.
    f = ComponentFactory(Registry())
    f.register("dropout", Dropout)
    first = f.config_adapter()
    f.registry.clear()
    f.register("linear", Linear)  # size back to 1, but a different class
    second = f.config_adapter()
    assert second is not first
    parsed = second.validate_python({"class_name": "linear", "in_features": 2, "out_features": 3})
    assert isinstance(parsed, Linear.Config)


def test_config_adapter_empty_registry_raises():
    with pytest.raises(ValueError, match="empty"):
        ComponentFactory(Registry()).config_adapter()


# --------------------------------------------------------------------------- #
# Scoped registries: co-resident libraries must not collide on tags.          #
# --------------------------------------------------------------------------- #
def test_scoped_factories_do_not_collide_on_tags():
    lib_a, lib_b = ComponentFactory(), ComponentFactory()

    class ABase(Component, factory=lib_a):
        class Config(Component.Config):
            pass

    class BBase(Component, factory=lib_b):
        class Config(Component.Config):
            pass

    class AThing(ABase):
        class Config(ABase.Config):
            class_name: Literal["scoped-thing"] = "scoped-thing"

    class BThing(BBase):  # the SAME tag in another library — no DuplicateTagError
        class Config(BBase.Config):
            class_name: Literal["scoped-thing"] = "scoped-thing"

    assert isinstance(lib_a.create({"class_name": "scoped-thing"}), AThing)
    assert isinstance(lib_b.create({"class_name": "scoped-thing"}), BThing)
    assert "scoped-thing" not in ComponentFactory.get().registry  # the global stays untouched


def test_scoped_container_builds_children_from_its_own_factory():
    lib = ComponentFactory()

    class SBase(Component, factory=lib):
        class Config(Component.Config):
            pass

    class SLeaf(SBase):
        class Config(SBase.Config):
            class_name: Literal["scoped-leaf"] = "scoped-leaf"

    class SBox(SBase):
        class Config(SBase.Config):
            class_name: Literal["scoped-box"] = "scoped-box"
            child: ChildConfig

        def _build_children(self, factory: ComponentFactory) -> None:
            self.child = factory.build(self.config.child)

    # The child spec resolves during validation, so it needs the scoped factory in context (the
    # global registry has never heard of 'scoped-leaf').
    cfg = SBox.Config.model_validate(
        {"class_name": "scoped-box", "child": {"class_name": "scoped-leaf"}},
        context={"factory": lib},
    )
    assert isinstance(cfg.child, SLeaf.Config)  # eagerly resolved to the concrete Config

    # Constructed DIRECTLY (not via the factory): ensure_built with no factory must fall back to the
    # family's scoped factory to BUILD the component.
    box = SBox(cfg)
    box.ensure_built()
    assert isinstance(box.child, SLeaf)
    assert "scoped-leaf" not in ComponentFactory.get().registry


# --------------------------------------------------------------------------- #
# ensure_built: idempotency, exception-safety, cycle detection.               #
# --------------------------------------------------------------------------- #
def test_ensure_built_is_idempotent(factory):
    head = factory.create(
        {"class_name": "sequential", "layers": [{"class_name": "dropout", "p": 0.1}]}
    )
    built_once = head.layers
    head.ensure_built(factory)  # second call must be a no-op, not rebuild
    assert head.layers is built_once


def test_ensure_built_is_retriable_after_a_failed_build():
    # A build that raises must leave the component un-built so a later call can retry — the built
    # flag is committed only after _build_children succeeds.
    class Flaky(Component):
        class Config(Component.Config):
            class_name: Literal["flaky"] = "flaky"

        def __init__(self, config):
            super().__init__(config)
            self.attempts = 0
            self.built = False

        def _build_children(self, factory):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("boom")
            self.built = True

    obj = Flaky(Flaky.Config())
    with pytest.raises(RuntimeError, match="boom"):
        obj.ensure_built()
    assert obj._children_built is False  # not falsely marked done
    obj.ensure_built()  # retry succeeds
    assert obj.built is True and obj.attempts == 2


def test_ensure_built_detects_a_cycle():
    # A component whose build re-enters its own ensure_built raises instead of recursing forever.
    class SelfReferential(Component):
        class Config(Component.Config):
            class_name: Literal["self-ref"] = "self-ref"

        def _build_children(self, factory):
            self.ensure_built(factory)  # re-entrant on the same instance

    obj = SelfReferential(SelfReferential.Config())
    with pytest.raises(RuntimeError, match="cyclic"):
        obj.ensure_built()
