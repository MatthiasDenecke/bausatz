"""Base class for config-driven classes — the ``Component`` family.

A ``Component`` declares a typed nested ``Config`` and self-registers by reading its Config's
``class_name`` discriminator, so it can be built from a plain dict/JSON spec via
``ComponentFactory``. Every concrete component pins its tag as a ``Literal`` field on its Config,
equal to the field's default:

    class MyComponent(Component):
        class Config(Component.Config):
            class_name: Literal["my_component"] = "my_component"
            ...other fields...

so the tag travels with the serialized config (``model_dump`` round-trips it) and is validated
like any other field. Registration happens at class-definition time into the process-global
``ComponentFactory`` — or, for a library that must coexist with others in one process, into a
SCOPED factory: ``class MyBase(Component, factory=my_factory)`` scopes the whole subtree, so tags
only need to be unique per library. Either way a component must be IMPORTED before the factory can
build it from a spec.

Container components declare their children with the natural ``Config`` type — no special field
type — and ``Component.Config`` resolves each child spec into its concrete ``Config`` subclass
during the parent's own validation:

    class Sequential(Component):
        class Config(Component.Config):
            class_name: Literal["sequential"] = "sequential"
            layers: list[Component.Config]      # or a family base, e.g. list[Retriever.Config]

        def _build_children(self, factory):
            self.layers = [factory.build(cfg) for cfg in self.config.layers]

So ``factory.validate`` validates a whole tree in one pass — a malformed grandchild fails there,
not later at build time — and the config stays typed rather than a bag of dicts. Typing children as
a family base (``list[Retriever.Config]``) additionally makes the resolver reject a child from a
different family.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from bausatz.component_factory import ComponentFactory

# --------------------------------------------------------------------------- #
# Component: base for config-driven classes.
# --------------------------------------------------------------------------- #
class Component:

    # The field every Config carries and the factory dispatches on. Defined once here and referenced
    # (never re-spelled as a literal) by the factory and the discriminated-union adapter.
    DISCRIMINATOR: ClassVar[str] = "class_name"

    class Config(BaseModel):
        name: str | None = None                     # shared field for every layer
        # Concrete subclasses pin the discriminator, e.g.:
        #     class_name: Literal["linear"] = "linear"

        model_config = ConfigDict(extra="forbid")  # reject unknown keys; inherited by subclasses

        @model_validator(mode="before")
        @classmethod
        def _resolve_children(cls, data: Any, info: ValidationInfo) -> Any:
            """Turn nested child specs into their concrete Configs, in the parent's validation pass.

            A field whose declared type is a ``Component.Config`` (or subclass) — directly or inside
            a ``list``/``dict`` — accepts a plain spec dict; here each such dict is looked up by its
            ``class_name`` in the registry (from the validation context's factory, else the global
            one) and replaced with the concrete Config. That is what lets ``factory.validate``
            validate a whole tree in one pass and keeps the tree typed, with no dedicated
            child-field type. An already-resolved Config, or a value that is not a spec, passes
            through untouched."""
            if not isinstance(data, Mapping):
                return data
            child_fields = _child_config_fields(cls)
            if not child_fields:
                return data
            patched = dict(data)
            for field_name, shape in child_fields.items():
                if field_name in patched:
                    patched[field_name] = _resolve_child(patched[field_name], shape, info)
            return patched

    # Narrowed in subclasses (`config: Linear.Config`) purely for type checkers.
    config: Component.Config

    # The factory this class family registers into — SCOPED REGISTRIES: a library defines its root
    # base with ``class MyBase(Component, factory=my_factory)`` and the whole subtree registers there,
    # so two libraries sharing one process never collide on a tag in the global table. ``None`` ⇒ the
    # process-global ``ComponentFactory.get()`` (single-library uses stay zero-config).
    _factory: ClassVar[ComponentFactory | None] = None

    def __init_subclass__(cls, factory: ComponentFactory | None = None, **kwargs: Any) -> None:
        from bausatz.component_factory import ComponentFactory
        super().__init_subclass__(**kwargs)
        if factory is not None:
            cls._factory = factory  # inherited by the subtree (a deeper subclass may re-scope)
        field = cls.Config.model_fields.get(cls.DISCRIMINATOR)
        if field is None:
            return  # no discriminator pinned -> abstract base, not registered
        if field.default is PydanticUndefined:
            tag = cls.__name__.lower()
            raise TypeError(
                f"{cls.__qualname__}.Config.{cls.DISCRIMINATOR} needs a default, e.g. "
                f'{cls.DISCRIMINATOR}: Literal["{tag}"] = "{tag}"'
            )
        (cls._factory or ComponentFactory.get()).register(field.default, cls)

    def __init__(self, config: Component.Config) -> None:
        self.config = config
        self._children_built = False  # flipped True by ensure_built once the children are built
        self._building = False        # guards against a cyclic graph re-entering its own build

    def _build_children(self, factory: ComponentFactory) -> None:
        """
        Hook for container components. No-op for leaf components.

        Override to construct nested components from ``self.config`` using the SAME
        factory, so the whole tree is built against one registry. Children arrive as
        already-validated ``Config`` objects, so pass them to ``factory.build`` — no
        re-validation:

            def _build_children(self, factory):
                self.children = [factory.build(cfg) for cfg in self.config.children]
        """

    def ensure_built(self, factory: ComponentFactory | None = None) -> None:
        """Build the children once and mark them built — the single owner of both the build and the
        ``_children_built`` flag, so they can't drift apart. ``ComponentFactory.build`` calls this with
        itself, so the tree builds against the one registry; a directly-constructed container (tests,
        ``Pipeline.from_stages``) calls it with no factory before first use and falls back to the
        family's scoped factory, then the global one.

        Idempotent: a second call is a no-op. Exception-safe: the built flag is set only after
        ``_build_children`` returns, so a failed build stays retriable rather than leaving a
        half-built object that claims to be done. A genuine cycle (a component whose build re-enters
        its own ``ensure_built``) raises rather than silently returning a partial object."""
        if self._children_built:
            return
        if self._building:
            raise RuntimeError(
                f"cyclic component graph: building {type(self).__name__} re-entered its own "
                f"ensure_built()"
            )
        from bausatz.component_factory import ComponentFactory

        # Fallback order: the caller's factory, the class family's scoped factory, the global.
        resolved = factory or type(self)._factory or ComponentFactory.get()
        self._building = True
        try:
            self._build_children(resolved)
        finally:
            self._building = False
        self._children_built = True  # commit only after a successful build

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the dict shape that ``ComponentFactory.create`` consumes.

        ``serialize_as_any`` makes nested child Configs (declared as a ``Component.Config`` base but
        holding a concrete subclass) emit their subclass fields, so the whole tree round-trips."""
        return self.config.model_dump(serialize_as_any=True)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config!r})"


# --------------------------------------------------------------------------- #
# Child-config resolution: which fields hold child specs, and how to resolve.
# --------------------------------------------------------------------------- #
# Per-config cache of {field_name: "one" | "list" | "dict"} for fields that hold child Configs.
_CHILD_FIELDS: dict[type, dict[str, str]] = {}


def _is_config_type(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, Component.Config)


def _classify_child_field(annotation: Any) -> str | None:
    """Return how a field holds child Configs — ``"one"`` / ``"list"`` / ``"dict"`` — or ``None``.

    Understands a bare ``Component.Config`` (subclass), ``list[...]``/``set``/``tuple`` and
    ``dict[K, ...]`` of them, and those wrapped in ``Optional`` / ``X | None``."""
    origin = get_origin(annotation)
    if origin is None:
        return "one" if _is_config_type(annotation) else None
    if origin is Union or origin is types.UnionType:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            shape = _classify_child_field(arg)
            if shape:
                return shape
        return None
    if origin in (list, set, frozenset, tuple):
        args = get_args(annotation)
        return "list" if args and _is_config_type(args[0]) else None
    if origin is dict:
        args = get_args(annotation)
        return "dict" if len(args) == 2 and _is_config_type(args[1]) else None
    return None


def _child_config_fields(config_cls: type) -> dict[str, str]:
    cached = _CHILD_FIELDS.get(config_cls)
    if cached is None:
        cached = {
            name: shape
            for name, field in config_cls.model_fields.items()
            if (shape := _classify_child_field(field.annotation)) is not None
        }
        _CHILD_FIELDS[config_cls] = cached
    return cached


def _resolve_one(value: Any, info: ValidationInfo) -> Any:
    """Resolve a single child spec dict into its concrete Config; pass anything else through."""
    if isinstance(value, Component.Config) or not isinstance(value, Mapping):
        return value
    if Component.DISCRIMINATOR not in value:
        return value  # let the field's own validation raise against the missing discriminator
    from bausatz.component_factory import ComponentFactory

    context = info.context or {}
    factory = context.get("factory") or ComponentFactory.get()
    klass = factory.registry.get(value[Component.DISCRIMINATOR])
    # Thread the same context down so deeper descendants resolve against the same factory.
    return klass.Config.model_validate(dict(value), context=info.context)


def _resolve_child(value: Any, shape: str, info: ValidationInfo) -> Any:
    if shape == "list" and isinstance(value, (list, tuple, set, frozenset)):
        return [_resolve_one(v, info) for v in value]
    if shape == "dict" and isinstance(value, Mapping):
        return {k: _resolve_one(v, info) for k, v in value.items()}
    if shape == "one":
        return _resolve_one(value, info)
    return value  # wrong container type for the shape: let field validation report it
