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

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError, PydanticUndefined
from pydantic_core.core_schema import ErrorType

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

        # The Component class that declared this Config, recorded by ``Component.__init_subclass__``.
        # Its factory (the family's scoped one, else the global) is where this Config's child specs
        # resolve when the validation context names no factory. A scoped family therefore validates
        # the same however it is entered: ``factory.validate``, ``factory.config_adapter()``, or
        # constructing the Config directly.
        _owner: ClassVar[type[Component] | None] = None

        @model_validator(mode="before")
        @classmethod
        def _resolve_children(cls, data: Any, info: ValidationInfo) -> Any:
            """Turn nested child specs into their concrete Configs, in the parent's validation pass.

            A field whose declared type is a ``Component.Config`` (or subclass), directly or inside
            a ``list``/``dict``, accepts a plain spec dict. Here each such dict is looked up by its
            ``class_name`` and replaced with the concrete Config. The lookup uses the validation
            context's factory, else this Config's home factory (see ``_owner``), and that factory
            is passed on so the whole tree resolves against one registry. That is what lets
            ``factory.validate`` validate a whole tree in one pass and keeps the tree typed, with no
            dedicated child-field type. An already-resolved Config, or a value that is not a spec,
            passes through untouched."""
            if not isinstance(data, Mapping):
                return data
            child_fields = _child_config_fields(cls)
            if not child_fields:
                return data
            context = _resolution_context(cls, info)
            errors: list[InitErrorDetails] = []
            patched = dict(data)
            for field_name, shape in child_fields.items():
                if field_name in patched:
                    patched[field_name] = _resolve_child(
                        patched[field_name], shape, (field_name,), context, errors
                    )
            if errors:
                # Raised from inside this validator, these line errors are merged into the outer
                # ValidationError at this Config's own location. Their locs are relative to this
                # Config (e.g. ('layers', 1, 'p')), so each one reads as the full path from the root.
                raise ValidationError.from_exception_data(cls.__name__, errors)
            return patched

    # Narrowed in subclasses (`config: Linear.Config`) purely for type checkers.
    config: Component.Config

    # The factory this class family registers into — SCOPED REGISTRIES: a library defines its root
    # base with ``class MyBase(Component, factory=my_factory)`` and the whole subtree registers there,
    # so two libraries sharing one process never collide on a tag in the global table. ``None`` ⇒ the
    # process-global ``ComponentFactory.get()`` (single-library uses stay zero-config).
    _factory: ClassVar[ComponentFactory | None] = None

    def __init_subclass__(cls, factory: ComponentFactory | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if factory is not None:
            cls._factory = factory  # inherited by the subtree (a deeper subclass may re-scope)
        if "_owner" not in vars(cls.Config):
            # The first component to declare this Config owns it. A subclass that inherits its
            # parent's Config leaves the parent as the owner.
            cls.Config._owner = cls
        field = cls.Config.model_fields.get(cls.DISCRIMINATOR)
        if field is None:
            return  # no discriminator pinned -> abstract base, not registered
        if field.default is PydanticUndefined:
            tag = cls.__name__.lower()
            raise TypeError(
                f"{cls.__qualname__}.Config.{cls.DISCRIMINATOR} needs a default, e.g. "
                f'{cls.DISCRIMINATOR}: Literal["{tag}"] = "{tag}"'
            )
        cls._home_factory().register(field.default, cls)

    @classmethod
    def _home_factory(cls) -> ComponentFactory:
        """The factory this class registers into: its family's scoped factory, else the global one.
        It is also the fallback for building children (``ensure_built``) and for resolving child
        specs (``Config._owner``) when no factory is given."""
        from bausatz.component_factory import ComponentFactory

        return cls._factory or ComponentFactory.get()

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
        # Fallback order: the caller's factory, the class family's scoped factory, the global.
        resolved = factory or type(self)._home_factory()
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


def _resolution_context(config_cls: type[Component.Config], info: ValidationInfo) -> Any:
    """The validation context to resolve ``config_cls``'s child specs with. This is the caller's
    context when it names a factory; otherwise it is a copy with ``factory`` set to the Config's
    home factory. A copy keeps the caller's object, and any other keys in it, untouched."""
    context = info.context
    if context is not None and context.get("factory") is not None:
        return context
    owner = config_cls._owner
    if owner is not None:
        factory = owner._home_factory()
    else:  # a Config no component declared: fall back to the global factory
        from bausatz.component_factory import ComponentFactory

        factory = ComponentFactory.get()
    return {**(context or {}), "factory": factory}


def _resolve_one(
    value: Any, loc: tuple[str | int, ...], context: Any, errors: list[InitErrorDetails]
) -> Any:
    """Resolve a single child spec dict into its concrete Config; pass anything else through.

    A child that fails validation is not raised on the spot. Its errors are collected into
    ``errors``, relocated under ``loc`` (the child's position in the parent), so every bad child of
    a Config is reported together, each at its full path."""
    if isinstance(value, Component.Config) or not isinstance(value, Mapping):
        return value
    if Component.DISCRIMINATOR not in value:
        return value  # let the field's own validation raise against the missing discriminator
    klass = context["factory"].registry.get(value[Component.DISCRIMINATOR])
    try:
        # Thread the same context down so deeper descendants resolve against the same factory.
        return klass.Config.model_validate(dict(value), context=context)
    except ValidationError as exc:
        errors.extend(_relocated(exc, loc))
        return value


def _resolve_child(
    value: Any,
    shape: str,
    loc: tuple[str | int, ...],
    context: Any,
    errors: list[InitErrorDetails],
) -> Any:
    if shape == "list" and isinstance(value, (list, tuple, set, frozenset)):
        return [_resolve_one(v, (*loc, i), context, errors) for i, v in enumerate(value)]
    if shape == "dict" and isinstance(value, Mapping):
        return {k: _resolve_one(v, (*loc, k), context, errors) for k, v in value.items()}
    if shape == "one":
        return _resolve_one(value, loc, context, errors)
    return value  # wrong container type for the shape: let field validation report it


# The error types pydantic-core can rebuild by name. Any other type came from a user's
# PydanticCustomError and is rebuilt as one.
_KNOWN_ERROR_TYPES = frozenset(get_args(ErrorType))


def _relocated(exc: ValidationError, prefix: tuple[str | int, ...]) -> list[InitErrorDetails]:
    """``exc``'s line errors, each with ``prefix`` prepended to its location and otherwise
    unchanged: same type, message, input and context."""
    relocated: list[InitErrorDetails] = []
    for err in exc.errors(include_url=False):
        loc = (*prefix, *err["loc"])
        ctx = err.get("ctx")
        if err["type"] in _KNOWN_ERROR_TYPES:
            detail = InitErrorDetails(type=err["type"], loc=loc, input=err["input"])
            if ctx is not None:
                detail["ctx"] = ctx
        else:
            custom = PydanticCustomError(err["type"], err["msg"], ctx)
            detail = InitErrorDetails(type=custom, loc=loc, input=err["input"])
        relocated.append(detail)
    return relocated
