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
from pydantic import Field, ValidationError, ValidationInfo, model_validator

from bausatz import Component, ComponentFactory, Registry
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
        layers: list[Component.Config]  # child specs, each resolved to its concrete Config eagerly

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


# A small family, scoped to its own factory, used to show that typing children as a family base
# (`list[_Retriever.Config]`) makes the resolver reject a child from another family. Defined at
# module scope so the forward reference resolves under `from __future__ import annotations`.
_family = ComponentFactory(Registry())


class _Retriever(Component, factory=_family):
    class Config(Component.Config):
        pass


class _Dense(_Retriever):
    class Config(_Retriever.Config):
        class_name: Literal["dense"] = "dense"


class _Pool(Component, factory=_family):  # deliberately NOT a _Retriever
    class Config(Component.Config):
        class_name: Literal["pool"] = "pool"


class _Ensemble(Component, factory=_family):
    class Config(Component.Config):
        class_name: Literal["ensemble"] = "ensemble"
        members: list[_Retriever.Config]


class Router(Component):
    """A container exercising the dict-shaped and optional child field types."""

    class Config(Component.Config):
        class_name: Literal["router"] = "router"
        routes: dict[str, Component.Config]         # dict-shaped children
        fallback: Component.Config | None = None    # optional child

    config: Router.Config

    def __init__(self, config: Router.Config) -> None:
        super().__init__(config)
        self.routes: dict[str, Component] = {}
        self.fallback: Component | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        self.routes = {k: factory.build(v) for k, v in self.config.routes.items()}
        self.fallback = factory.build(self.config.fallback) if self.config.fallback else None


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


def test_family_typed_children_reject_a_foreign_child():
    # Typing children as a family base (list[_Retriever.Config]) makes the resolver enforce the
    # family: a member resolves, a component from another family is rejected at validation.
    ok = _family.validate({"class_name": "ensemble", "members": [{"class_name": "dense"}]})
    assert isinstance(ok.members[0], _Dense.Config)
    with pytest.raises(ValidationError):
        _family.validate({"class_name": "ensemble", "members": [{"class_name": "pool"}]})


def test_dict_and_optional_children_resolve(factory):
    factory.register("router", Router)
    cfg = factory.validate(
        {
            "class_name": "router",
            "routes": {
                "a": {"class_name": "linear", "in_features": 2, "out_features": 1},
                "b": {"class_name": "dropout", "p": 0.2},
            },
            "fallback": {"class_name": "dropout", "p": 0.0},
        }
    )
    assert isinstance(cfg.routes["a"], Linear.Config)
    assert isinstance(cfg.routes["b"], Dropout.Config)
    assert isinstance(cfg.fallback, Dropout.Config)

    router = factory.build(cfg)
    assert isinstance(router.routes["a"], Linear) and router.fallback is not None
    assert factory.create(router.to_dict()).config == router.config  # dict/optional round-trip

    # An omitted optional child stays None.
    assert factory.validate({"class_name": "router", "routes": {}}).fallback is None


def test_wrong_container_type_for_children_is_rejected(factory):
    # layers is list-shaped; handing it a dict is left for field validation to reject.
    with pytest.raises(ValidationError):
        factory.validate(
            {"class_name": "sequential", "layers": {"x": {"class_name": "dropout", "p": 0.1}}}
        )


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


def _error_locs(exc_info: pytest.ExceptionInfo[ValidationError]) -> list[tuple]:
    return [err["loc"] for err in exc_info.value.errors()]


def test_child_validation_errors_carry_their_full_path(factory):
    # A child's error is reported at its path from the root, not relative to the child.
    with pytest.raises(ValidationError) as exc_info:
        factory.validate(
            {
                "class_name": "sequential",
                "layers": [{"class_name": "linear", "in_features": 4, "out_features": 2},
                           {"class_name": "dropout", "p": 5.0}],
            }
        )
    assert _error_locs(exc_info) == [("layers", 1, "p")]
    assert exc_info.value.errors()[0]["type"] == "less_than_equal"


def test_grandchild_validation_errors_carry_their_full_path(factory):
    with pytest.raises(ValidationError) as exc_info:
        factory.validate(
            {
                "class_name": "sequential",
                "layers": [{"class_name": "sequential",
                            "layers": [{"class_name": "dropout", "p": 5.0}]}],
            }
        )
    assert _error_locs(exc_info) == [("layers", 0, "layers", 0, "p")]


def test_every_bad_child_is_reported_with_dict_and_optional_paths(factory):
    # All failing children are reported together (not just the first), under dict keys and
    # optional single-child fields as well as list indices.
    factory.register("router", Router)
    with pytest.raises(ValidationError) as exc_info:
        factory.validate(
            {
                "class_name": "router",
                "routes": {"a": {"class_name": "dropout", "p": 0.1},
                           "b": {"class_name": "linear", "in_features": 4, "bogus": 1}},
                "fallback": {"class_name": "dropout", "p": -1.0},
            }
        )
    assert sorted(_error_locs(exc_info), key=str) == sorted(
        [("routes", "b", "out_features"), ("routes", "b", "bogus"), ("fallback", "p")], key=str
    )


def test_custom_child_error_type_is_preserved(factory):
    from pydantic import field_validator
    from pydantic_core import PydanticCustomError

    class Picky(Component):
        class Config(Component.Config):
            class_name: Literal["picky"] = "picky"
            level: int = 0

            @field_validator("level")
            @classmethod
            def _odd_only(cls, v: int) -> int:
                if v % 2 == 0:
                    raise PydanticCustomError("not_odd", "level {level} is not odd", {"level": v})
                return v

    factory.register("picky", Picky)
    with pytest.raises(ValidationError) as exc_info:
        factory.validate({"class_name": "sequential", "layers": [{"class_name": "picky", "level": 2}]})
    [err] = exc_info.value.errors()
    assert err["loc"] == ("layers", 0, "level")
    assert err["type"] == "not_odd" and err["msg"] == "level 2 is not odd"
    assert err["ctx"] == {"level": 2}


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


def test_registry_unhashable_tag_is_unknown():
    # A malformed spec may carry any value as its tag; an unhashable one can't be registered, so it
    # is reported as unknown rather than leaking a TypeError from the dict lookup.
    reg = Registry()
    reg.register("x", Linear)
    with pytest.raises(UnknownTagError):
        reg.get(["x"])
    assert ["x"] not in reg


def test_unhashable_class_name_is_an_unknown_tag(factory):
    with pytest.raises(UnknownTagError):
        factory.validate({"class_name": ["linear"]})  # root spec
    with pytest.raises(UnknownTagError):
        factory.validate({"class_name": "sequential", "layers": [{"class_name": {}}]})  # child spec


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
            child: SBase.Config  # a family-base type: the resolver rejects a non-SBase child

        def _build_children(self, factory: ComponentFactory) -> None:
            self.child = factory.build(self.config.child)

    # The child spec resolves during validation, against the scoped factory (the global registry has
    # never heard of 'scoped-leaf'). It comes from the context when given, and otherwise from the
    # family SBox.Config belongs to, so every way of validating the Config resolves the child.
    spec = {"class_name": "scoped-box", "child": {"class_name": "scoped-leaf"}}
    cfg = SBox.Config.model_validate(spec, context={"factory": lib})
    assert isinstance(cfg.child, SLeaf.Config)  # eagerly resolved to the concrete Config
    assert isinstance(SBox.Config.model_validate(spec).child, SLeaf.Config)  # no context
    assert isinstance(SBox.Config(child={"class_name": "scoped-leaf"}).child, SLeaf.Config)
    assert isinstance(lib.config_adapter().validate_python(spec).child, SLeaf.Config)

    # Constructed DIRECTLY (not via the factory): ensure_built with no factory must fall back to the
    # family's scoped factory to BUILD the component.
    box = SBox(cfg)
    box.ensure_built()
    assert isinstance(box.child, SLeaf)
    assert "scoped-leaf" not in ComponentFactory.get().registry


def test_child_specs_resolve_against_the_parents_factory_all_the_way_down():
    # A whole tree resolves against one registry: the factory a Config falls back to is passed on
    # to its descendants, so a grandchild resolves where the root did.
    lib = ComponentFactory()

    class TBase(Component, factory=lib):
        class Config(Component.Config):
            pass

    class TLeaf(TBase):
        class Config(TBase.Config):
            class_name: Literal["tree-leaf"] = "tree-leaf"

    class TNode(TBase):
        class Config(TBase.Config):
            class_name: Literal["tree-node"] = "tree-node"
            kids: list[TBase.Config] = []

    cfg = TNode.Config(kids=[{"class_name": "tree-node", "kids": [{"class_name": "tree-leaf"}]}])
    assert isinstance(cfg.kids[0].kids[0], TLeaf.Config)


def test_caller_context_is_passed_through_to_children():
    # Extra keys in the caller's context reach the child validators, and the caller's own
    # context object is not modified when the factory is filled in.
    seen: list[object] = []

    class Probe(Component):
        class Config(Component.Config):
            class_name: Literal["ctx-probe"] = "ctx-probe"

            @model_validator(mode="after")
            def _record(self, info: ValidationInfo) -> Probe.Config:
                seen.append((info.context or {}).get("tenant"))
                return self

    class Holder(Component):
        class Config(Component.Config):
            class_name: Literal["ctx-holder"] = "ctx-holder"
            child: Component.Config

    context = {"tenant": "acme"}
    Holder.Config.model_validate(
        {"class_name": "ctx-holder", "child": {"class_name": "ctx-probe"}}, context=context
    )
    assert seen == ["acme"]
    assert context == {"tenant": "acme"}  # no "factory" key leaked into the caller's dict

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
