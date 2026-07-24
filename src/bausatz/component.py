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

Container components declare their children with ``ChildConfig`` rather than a raw ``dict``:

    class Sequential(Component):
        class Config(Component.Config):
            class_name: Literal["sequential"] = "sequential"
            layers: list[ChildConfig]   # each element validates into its concrete Config

        def _build_children(self, factory):
            self.layers = [factory.build(cfg) for cfg in self.config.layers]

``ChildConfig`` resolves each child spec into its concrete ``Config`` subclass *eagerly*, during
the parent's validation, so a malformed grandchild is caught by ``factory.validate`` — not deferred
to build time — and the whole config tree is typed rather than a bag of dicts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, SerializeAsAny, ValidationInfo
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
        already-validated ``Config`` objects (see ``ChildConfig``), so pass them to
        ``factory.build`` — no re-validation:

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
        """Serialize back to the dict shape that ``ComponentFactory.create`` consumes."""
        return self.config.model_dump()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config!r})"


# --------------------------------------------------------------------------- #
# ChildConfig: a nested child spec, resolved to its concrete Config eagerly.
# --------------------------------------------------------------------------- #
def _resolve_child_config(value: Any, info: ValidationInfo) -> Any:
    """Turn a child spec ``{"class_name": ..., ...}`` into its concrete ``Config`` subclass.

    Runs during the *parent's* validation, so the whole tree is validated in one pass. The registry
    is taken from the validation context (``model_validate(spec, context={"factory": f})``, which
    ``ComponentFactory.validate``/``create`` seed with themselves) so a scoped factory resolves its
    own tags; absent a context it falls back to the process-global factory. An already-resolved
    ``Config`` instance passes straight through."""
    if isinstance(value, Component.Config):
        return value
    if not isinstance(value, Mapping) or Component.DISCRIMINATOR not in value:
        raise ValueError(
            f"child spec must be a mapping with a {Component.DISCRIMINATOR!r} key: {value!r}"
        )
    from bausatz.component_factory import ComponentFactory

    context = info.context or {}
    factory = context.get("factory") or ComponentFactory.get()
    klass = factory.registry.get(value[Component.DISCRIMINATOR])
    # Thread the same context down so deeper descendants resolve against the same factory.
    return klass.Config.model_validate(dict(value), context=info.context)


# A container field type: ``layers: list[ChildConfig]`` / ``child: ChildConfig``. ``SerializeAsAny``
# makes ``model_dump`` emit the concrete subclass's fields (not just the ``Component.Config`` base),
# so ``to_dict`` round-trips the full tree; ``BeforeValidator`` does the dict -> concrete-Config
# resolution above.
ChildConfig = Annotated[SerializeAsAny[Component.Config], BeforeValidator(_resolve_child_config)]
