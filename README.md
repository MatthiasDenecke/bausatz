# bausatz

[![tests](https://github.com/MatthiasDenecke/bausatz/actions/workflows/codecov.yml/badge.svg)](https://github.com/MatthiasDenecke/bausatz/actions/workflows/codecov.yml)
[![codecov](https://codecov.io/gh/MatthiasDenecke/bausatz/branch/main/graph/badge.svg)](https://codecov.io/gh/MatthiasDenecke/bausatz)
[![PyPI](https://img.shields.io/pypi/v/bausatz)](https://pypi.org/project/bausatz/)
[![python](https://img.shields.io/pypi/pyversions/bausatz)](https://pypi.org/project/bausatz/)

*German: “construction kit.”*

**Config-driven component trees for Python** — pydantic-validated classes that self-register by a
`class_name` tag and assemble recursively from plain dict/JSON/YAML specs. Components can be pure
Python or **backed by C++** (via pybind11) — same config, same factory, either way. See
[Native (C++) components](#native-c-components).

```python
from typing import Literal
from bausatz import Component, ComponentFactory

class Retriever(Component):                      # an abstract family (no tag -> not registered)
    class Config(Component.Config):
        pass

class Dense(Retriever):                          # a concrete member, registered as "dense"
    class Config(Retriever.Config):
        class_name: Literal["dense"] = "dense"
        top_k: int = 50

class Pipeline(Component):                       # a container builds its children through the factory
    class Config(Component.Config):
        class_name: Literal["pipeline"] = "pipeline"
        retrievers: list[Retriever.Config]       # child specs -> concrete Configs, validated eagerly

    def _build_children(self, factory: ComponentFactory) -> None:
        self.retrievers = [factory.build(cfg) for cfg in self.config.retrievers]

pipeline = ComponentFactory.get().create(
    {"class_name": "pipeline", "retrievers": [{"class_name": "dense", "top_k": 20}]}
)
```

A whole system becomes a spec: swap implementations by editing config, not code. Unknown keys are
rejected and fields are validated — for the whole tree in one pass, since a child field typed as a
`Config` (here `list[Retriever.Config]`) resolves and validates its nested specs during the parent's
validation, and rejects a child from another family — and `to_dict()` round-trips every component
back to the spec that built it.

## The pieces

| piece | job |
|---|---|
| `Component` | base for buildable classes — declares a typed nested `Config` (pydantic) and self-registers at class-definition time by its `class_name` tag |
| `ComponentFactory` | `validate(spec)` → a typed `Config` (children resolved), `build(config)` → the instance, and `create = build∘validate` (plus typed `create_as` / `build_as` and `create_many`), recursing into container children; `config_adapter()` builds a pydantic **discriminated union** over every registered Config for one-pass validation of whole trees |
| `Registry` | the tag → class table; duplicate tags fail loudly |
| `CppComponent` | a `Component` whose compute is a native (C++) component built from the validated config — see [Native (C++) components](#native-c-components) |

A container's child fields are typed as a `Config` (`list[Retriever.Config]`, `dict[str, Component.Config]`, an optional one, …); `Component.Config` resolves each nested spec into its concrete Config eagerly, in the parent's validation pass — no dedicated child-field type to import.

## Scoped registries

Registration defaults to a process-global factory (zero config for a single library). Libraries
that must **coexist in one process** scope their family instead — tags then only need to be unique
per library:

```python
my_factory = ComponentFactory()

class MyBase(Component, factory=my_factory):     # the whole subtree registers into my_factory
    class Config(Component.Config):
        pass
```

## Native (C++) components

A component's behavior can be implemented in C++. `CppComponent` is a `Component` whose compute is a
native object (typically a [pybind11](https://pybind11.readthedocs.io)-exposed class): the pydantic
`Config` still does validation, tagging, registration and `to_dict`, while the native component —
built from the *validated* config — does the work. It's **declarative** — name the native class with
`component=` and the framework marshals the config fields in as keyword arguments, so a C++-backed
component needs no more code than a pure-Python one:

```python
class Affine(CppComponent, component=AffineKernel):   # AffineKernel is a pybind11 class
    class Config(Component.Config):
        class_name: Literal["affine"] = "affine"
        scale: float = 1.0
        bias: float = 0.0
    def forward(self, x): return self.component.forward(x)   # self.component is built lazily
```

`AffineKernel(scale=..., bias=...)` is constructed for you from the config fields; override
`_create_component` only when construction is custom (e.g. a container that builds child components).
The library ships a C++ base `bausatz::Component` (header-only) so native components share a
polymorphic root — enough for a C++ container to run a whole subtree natively. A downstream extension
finds the
headers with `bausatz.get_include()`. See [`examples/bausatz`](examples/bausatz) for a worked, compiled
example (a declarative leaf and a native C++ composite).

## Design notes

- **The tag lives on the Config** (`class_name: Literal["dense"] = "dense"`), so it serializes with
  the config and validates like any other field.
- **Children are typed, not `dict`**: typing them as a `Config` validates the whole config tree in
  one eager pass (a malformed grandchild fails `validate`, not later at build time), keeps the config
  self-describing, and — with a family base like `list[Retriever.Config]` — rejects a foreign child.
- **Containers own their children**: `_build_children(factory)` turns the child Configs into
  components through the *same* factory, so a whole tree assembles against one registry in one
  recursive pass.
- **Import = register**: a component must be imported before a spec can name it (package
  `__init__`s that import their members are the idiomatic registration point).
- Zero dependencies beyond **pydantic v2** (≥ 2.7). Python ≥ 3.12. Fully typed (`py.typed`).

## Install

```bash
pip install bausatz
```

MIT licensed.
