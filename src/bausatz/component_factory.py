"""Tag-dispatched factory that turns a serialized spec dict into a ``Component``.

The framework is three small modules, each with one job:

  Registry          (registry.py)         the tag -> class mapping, a pure lookup table
  Component         (component.py)        base for buildable classes; declares a typed nested
                                          Config and self-registers by its ``class_name`` tag
  ComponentFactory  (this module)         validate a spec into its Config, construct the
                                          Component, recursing into nested children

A ``ComponentFactory`` holds a ``Registry`` and is also reachable as a process-global singleton
via ``ComponentFactory.get()`` — which is where ``Component`` subclasses self-register on import.
``Component`` only references ``ComponentFactory`` inside a ``TYPE_CHECKING`` annotation (plus a
lazy import in ``__init_subclass__`` / ``ensure_built`` / the child-config resolver), which
avoids the import cycle between the two modules.

Construction is two steps, exposed separately:
  ``validate(spec)`` -> a typed ``Config`` (recursively validated, children resolved to their
                        concrete Configs), and
  ``build(config)``  -> the ``Component`` for an already-validated Config.
``create(spec)`` is just ``build(validate(spec))``. Splitting them lets a caller validate a whole
config tree up front (e.g. to surface errors) without constructing anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from operator import or_
from typing import Annotated, Any, TypeVar

from pydantic import Field, TypeAdapter

from .component import Component
from .registry import Registry

_C = TypeVar("_C", bound=Component)


# -------------------------------------------------------------------------------------- #
# ComponentFactory: construction. Holds a Registry; turns specs into Component objects.  #
# -------------------------------------------------------------------------------------- #
class ComponentFactory:

    @staticmethod
    def get() -> ComponentFactory:
        return _component_factory

    def __init__(self, registry: Registry[Component] | None = None) -> None:
        self.registry: Registry[Component] = registry if registry is not None else Registry[Component]()

        self._adapter_cache: tuple[int, TypeAdapter[Any]] | None = None

    def validate(self, spec: Mapping[str, Any]) -> Component.Config:
        """
        Validate a spec into its Config without constructing the Component.

        Validation is recursive: any child-config fields (children typed as a ``Config``) resolve
        their child specs into concrete Configs in the same pass, against THIS factory (seeded via
        the validation context), so scoped registries resolve their own tags.
        """
        if not isinstance(spec, Mapping) or Component.DISCRIMINATOR not in spec:
            raise ValueError(
                f"spec must be a mapping with a {Component.DISCRIMINATOR!r} key: {spec!r}"
            )
        klass = self.registry.get(spec[Component.DISCRIMINATOR])
        return klass.Config.model_validate(dict(spec), context={"factory": self})

    def build(self, config: Component.Config) -> Component:
        """Construct the ``Component`` for an already-validated ``Config`` (no re-validation) and
        build its children against THIS factory. This is the step containers call in
        ``_build_children`` for each of their child Configs."""
        tag = getattr(config, Component.DISCRIMINATOR, None)
        if tag is None:
            raise ValueError(
                f"{type(config).__name__} has no {Component.DISCRIMINATOR!r} field; not buildable"
            )
        klass = self.registry.get(tag)
        obj = klass(config)
        obj.ensure_built(self)  # build the tree against THIS factory + mark built (one owner)
        return obj

    def build_as(self, config: Component.Config, expected: type[_C]) -> _C:
        """``build`` plus an ``isinstance`` assertion — the typed way for a container to build a
        child Config while pinning the component subtype it expects (e.g. a ``Chunk`` stage's
        chunker must be a ``Chunker``)."""
        obj = self.build(config)
        if not isinstance(obj, expected):
            raise TypeError(
                f"{getattr(config, Component.DISCRIMINATOR, None)!r} is a {type(obj).__name__}, "
                f"not a {expected.__name__}"
            )
        return obj

    def create(self, spec: Mapping[str, Any]) -> Component:
        """Validate a spec and build the ``Component`` — ``build(validate(spec))``."""
        return self.build(self.validate(spec))

    def create_as(self, spec: Mapping[str, Any], expected: type[_C]) -> _C:
        """Validate a spec, build it, and assert it is an ``expected`` instance — the spec-level
        counterpart of ``build_as``."""
        return self.build_as(self.validate(spec), expected)

    def create_many(self, specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Component]:
        """
        Build a dict of named layers, e.g. {"encoder": {...}, "head": {...}}.
        """
        return {name: self.create(spec) for name, spec in specs.items()}

    def config_adapter(self) -> TypeAdapter[Any]:
        """
        A Pydantic discriminated union over every registered Config.

        Validates a whole nested structure in one pass, with Pydantic routing each
        node by `class_name`. Rebuilt only when the registry changes (keyed on its
        monotonic version, so a clear + re-register can't serve a stale adapter).

        Nested child specs resolve against each Config's home factory (its family's
        scoped factory, else the global one), so a scoped family needs no extra
        arguments. A class registered here by hand whose home is a *different* factory
        resolves its children there unless you pass ``context={"factory": self}``,
        which is what ``validate()`` does.
        """
        version = self.registry.version
        if self._adapter_cache is not None and self._adapter_cache[0] == version:
            return self._adapter_cache[1]
        configs = [cls.Config for cls in self.registry.classes()]
        if not configs:
            raise ValueError("registry is empty; nothing to validate against")
        # Discriminated even when only one Config is registered. Every Config defaults its tag, so a
        # plain single-model adapter would accept a spec with no class_name that the union rejects
        # as soon as a second class registers.
        union = reduce(or_, configs)  # ConfigA | ConfigB | ..., or the lone Config
        adapter: TypeAdapter[Any] = TypeAdapter(
            Annotated[union, Field(discriminator=Component.DISCRIMINATOR)]
        )
        self._adapter_cache = (version, adapter)
        return adapter

    def register(self, tag: str, cls: type[Component]) -> None:
        self.registry.register(tag, cls)


_component_factory = ComponentFactory()
