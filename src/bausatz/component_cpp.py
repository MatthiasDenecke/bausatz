"""Bridge for components whose behavior is implemented natively — typically a C++ class exposed
with pybind11, though any Python-callable class works.

The division of labor:

  * The pydantic ``Config`` keeps everything that makes a component config-driven — field
    validation, the ``class_name`` tag, factory registration, ``to_dict`` round-trip — all in
    Python, where pydantic lives. The native side never parses or validates a spec.
  * The native **component** owns the compute. It is constructed from the *already-validated*
    config.

The construction is declarative: a subclass names the native class it wraps, and the framework
builds it by marshalling the validated config fields in as keyword arguments — so a C++-backed leaf
needs no more code than a pure-Python component:

    class Affine(CppComponent, component=AffineKernel):   # AffineKernel is a pybind11 class
        class Config(Component.Config):
            class_name: Literal["affine"] = "affine"
            scale: float = 1.0
            bias: float = 0.0

        def forward(self, x):
            return self.component.forward(x)              # straight into C++

The default ``_create_component`` calls ``component_cls(**fields)``, where ``fields`` is the
validated config minus the framework-only ``class_name`` (the field names must match the native
constructor's keyword arguments — including ``name``, which the native component takes at
construction and keeps immutably). Components that need custom construction — notably containers that
build child components — override ``_create_component`` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic_core import PydanticUndefined

from bausatz.component import Component

_UNSET: Any = object()  # a private sentinel, distinct from a legitimate ``None`` component


class CppComponent(Component):
    """A ``Component`` whose behavior is provided by a native component object.

    Declare the native class with ``class Foo(CppComponent, component=NativeFoo)`` (or set
    ``component_cls`` directly). The native object is built lazily on first access to ``component``
    and cached — so a ``CppComponent`` is usable whether it was assembled by the factory or
    constructed directly.

    A concrete ``CppComponent`` must be buildable: it either declares ``component=`` /
    ``component_cls`` or overrides ``_create_component`` — otherwise its class definition raises.
    """

    # The native class this wraps — usually a pybind11 class deriving from ``bausatz::Component``.
    component_cls: ClassVar[Callable[..., Any] | None] = None

    def __init_subclass__(cls, component: Callable[..., Any] | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if component is not None:
            cls.component_cls = component
        # A concrete (registered) CppComponent must be able to build its native object: either it
        # has a component_cls, or it overrides _create_component. Abstract intermediates (no pinned
        # class_name) are exempt.
        field = cls.Config.model_fields.get(Component.DISCRIMINATOR)
        is_concrete = field is not None and field.default is not PydanticUndefined
        overrides_create = cls._create_component is not CppComponent._create_component
        if is_concrete and cls.component_cls is None and not overrides_create:
            raise TypeError(
                f"{cls.__qualname__} is a concrete CppComponent with no native component to build: "
                f"declare `component=<class>` (or set component_cls), or override _create_component()."
            )

    def __init__(self, config: Component.Config) -> None:
        super().__init__(config)
        self._component: Any = _UNSET  # built on first ``component`` access

    def _component_fields(self) -> dict[str, Any]:
        """The validated config fields passed as keyword args to ``component_cls``. Defaults to the
        whole config minus the framework-only ``class_name``; override to rename, filter, or remap
        fields for the native constructor.

        ``name`` is kept: the native component takes its name at construction (``bausatz::Component``
        makes it immutable). Since the config's ``name`` is ``str | None`` while the native side
        expects a string, ``None`` is normalized to ``""``."""
        fields = self.config.model_dump(exclude={Component.DISCRIMINATOR})
        if fields.get("name") is None:
            fields["name"] = ""
        return fields

    def _create_component(self) -> Any:
        """Construct the native component from the config —
        ``component_cls(**self._component_fields())``. Override for custom construction, e.g. a
        container that builds its child components."""
        if self.component_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no component_cls; declare `component=` or override "
                f"_create_component()."
            )
        return self.component_cls(**self._component_fields())

    @property
    def component(self) -> Any:
        """The native component, built from the config on first access and cached thereafter.

        The children are built first (``ensure_built()``, a no-op once the factory has built them).
        A container constructed directly, not through the factory, therefore still sees its children
        when ``_create_component`` runs, instead of caching a native object built from none. As a
        consequence ``_build_children`` must not read ``self.component``: the native object depends
        on the children, not the other way round."""
        if self._component is _UNSET:
            self.ensure_built()
            self._component = self._create_component()
        return self._component
