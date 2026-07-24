# Example: C++ classes as bausatz components (pybind11)

Drives compiled C++ classes from Python as ordinary config-driven `Component`s. Python keeps the
config/validation/registry/factory; C++ keeps the compute. Two shapes:

```
# Composition — Python owns the tree, a C++ leaf does the compute:
spec ─▶ ComponentFactory ─▶ Affine(CppComponent)                      ← Python: pydantic Config
                                │  _make_engine: AffineConfig(scale, bias)   ← config struct
                                ▼
                           AffineKernel : Transform : bausatz::Component ← C++ (extension _affine)

# Native subtree — a C++ composite runs the whole chain, no per-stage bounce to Python:
spec ─▶ ComponentFactory ─▶ NativeSequential(CppComponent)
                                │  _build_children → child Affines
                                │  _make_engine: Sequential([child.engine, ...])
                                ▼
                           Sequential : Transform : bausatz::Component  ← chains children in C++
```

## The library base vs. the app's compute interface

`bausatz::Component` is part of the **bausatz library**, not this example — the package ships the
header and a downstream extension finds it with `bausatz.get_include()`:

```python
import bausatz
Pybind11Extension(..., include_dirs=[bausatz.get_include()])   # #include <bausatz/component.hpp>
```

It is header-only (links nothing) and declares **no compute method** — what an engine computes is
application-specific. So this example defines its *own* interface, `Transform : bausatz::Component`,
and that is where `forward` lives. `AffineKernel` (a leaf) and `Sequential` (a composite) derive
from `Transform`; `Sequential` holds its children as `std::shared_ptr<Transform>` and runs the whole
chain in C++ from one `forward` call. The library base carries only shared plumbing (e.g. a `name`,
mirroring `Component.Config.name`) and the `shared_ptr`-friendly polymorphic root.

## Files

| file | role |
|---|---|
| `src/affine.cpp` | app interface `Transform`, `AffineConfig` struct, `AffineKernel` (leaf), `Sequential` (native composite), and the `PYBIND11_MODULE` |
| `bausatz_affine/__init__.py` | `Affine` (config-struct passing) and `NativeSequential` (its engine is the C++ `Sequential`) |
| `setup.py` | builds with `Pybind11Extension` + `build_ext`, `include_dirs=[bausatz.get_include()]` |
| `test_affine.py` | config-struct passing, `forward` on `Transform` (not the base), the `name` plumbing, native chaining + nesting, GIL/STL batch, C++→Python exceptions |

The C++ base and the pybind helpers live in the library: `src/bausatz/include/bausatz/component.hpp`
and `.../pybind.hpp`.

## Build & run

```bash
pip install pybind11                         # build-time dependency
python setup.py build_ext --inplace          # compiles bausatz_affine/_affine*.so
python -m pytest test_affine.py
```

Or install it as a package: `pip install --no-build-isolation .`

## What pybind11 is doing for you

- **`bausatz::bind_component<AffineKernel, Transform>(m, "AffineKernel")`** (from `<bausatz/pybind.hpp>`)
  — the library's helper for `py::class_<T, Base, std::shared_ptr<T>>`: records the base and a
  shared-ownership holder so instances upcast to `shared_ptr<Transform>` when handed to `Sequential`.
- **Type casters (`<pybind11/stl.h>`)** — scalars cross as plain numbers; `list[Transform]` ⇄
  `std::vector<std::shared_ptr<Transform>>` and `list[float]` ⇄ `std::vector<double>` are automatic.
- **`py::call_guard<py::gil_scoped_release>()`** — releases the GIL around the C++ batch loop.
- **Exception translation** — `std::invalid_argument` surfaces in Python as `ValueError`, next to
  pydantic's `ValidationError`.

## How the config crosses the boundary

`Affine` uses **config-struct passing**: `_make_engine` builds a typed C++ `AffineConfig` from the
validated pydantic config and hands that one object to the kernel — the C++ side never sees pydantic.
For a handful of fields you can also just unpack them into positional constructor args; for a large
or dynamic config, pass `self.config.model_dump()` as a `dict`. In every case the pydantic `Config`
is the validated source of truth in Python, and only plain data crosses into C++.

## When to use which shape

- **Composition** (`Affine`) is the default: least friction, and the Python tree stays in charge.
- **Native subtree** (`NativeSequential`) pays off when the compute composition itself should run in
  C++ (tight loops over many stages, no per-node GIL crossing). Its children must themselves be
  C++-backed — their `engine` is a `Transform`. Mixing in a pure-Python child won't work, because the
  C++ `Sequential` only holds `Transform`s.
- **Trampolines** (`PYBIND11_OVERRIDE`, not shown) are the third shape — for overriding a C++ virtual
  *in Python*, or a C++ driver calling back into Python stages.
