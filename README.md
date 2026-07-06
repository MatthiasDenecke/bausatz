# bausatz

[![tests](https://github.com/MatthiasDenecke/bausatz/actions/workflows/codecov.yml/badge.svg)](https://github.com/MatthiasDenecke/bausatz/actions/workflows/codecov.yml)
[![codecov](https://codecov.io/gh/MatthiasDenecke/bausatz/branch/main/graph/badge.svg)](https://codecov.io/gh/MatthiasDenecke/bausatz)
[![PyPI](https://img.shields.io/pypi/v/bausatz)](https://pypi.org/project/bausatz/)
[![python](https://img.shields.io/pypi/pyversions/bausatz)](https://pypi.org/project/bausatz/)

*German: “construction kit.”*

**Config-driven component trees for Python** — pydantic-validated classes that self-register by a
`class_name` tag and assemble recursively from plain dict/JSON/YAML specs.

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
        retrievers: list[dict]

    def _build_children(self, factory: ComponentFactory) -> None:
        self.retrievers = [factory.create_as(spec, Retriever) for spec in self.config.retrievers]

pipeline = ComponentFactory.get().create(
    {"class_name": "pipeline", "retrievers": [{"class_name": "dense", "top_k": 20}]}
)
```

A whole system becomes a spec: swap implementations by editing config, not code. Unknown keys are
rejected, fields are validated, and `to_json()` round-trips every component back to the spec that
built it.

## The pieces

| piece | job |
|---|---|
| `Component` | base for buildable classes — declares a typed nested `Config` (pydantic) and self-registers at class-definition time by its `class_name` tag |
| `ComponentFactory` | turns a spec dict into an instance (`create` / typed `create_as` / `create_many`), recursing into container children; `config_adapter()` builds a pydantic **discriminated union** over every registered Config for one-pass validation of whole trees |
| `Registry` | the tag → class table; duplicate tags fail loudly |

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

## Design notes

- **The tag lives on the Config** (`class_name: Literal["dense"] = "dense"`), so it serializes with
  the config and validates like any other field.
- **Containers own their children**: `_build_children(factory)` builds nested components through
  the *same* factory, so a whole tree assembles against one registry in one recursive pass.
- **Import = register**: a component must be imported before a spec can name it (package
  `__init__`s that import their members are the idiomatic registration point).
- Zero dependencies beyond **pydantic v2**. Python ≥ 3.12. Fully typed (`py.typed`).

## Install

```bash
pip install bausatz
```

MIT licensed.
