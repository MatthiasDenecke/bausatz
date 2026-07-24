# Example: a C++ class as a bausatz component (pybind11)

Drives a compiled C++ class from Python as an ordinary config-driven `Component`. It shows the
recommended **composition** bridge: Python keeps the config/validation/registry/factory; C++ keeps
the compute.

```
spec dict ─▶ ComponentFactory ─▶ Affine(CppComponent)      ← Python: pydantic Config, tag "affine"
                                     │  _make_engine(scale, bias)
                                     ▼
                                AffineKernel(...)            ← C++: pybind11 extension (_affine)
```

## Files

| file | role |
|---|---|
| `src/affine.cpp` | the C++ `AffineKernel` + the `PYBIND11_MODULE` that exposes it — the entire bridge |
| `bausatz_affine/__init__.py` | `Affine(CppComponent)`: the pydantic `Config`, `_make_engine` (calls the C++ ctor), and `forward` delegating to the kernel |
| `setup.py` | builds the extension with `Pybind11Extension` + `build_ext` |
| `test_affine.py` | spec → factory → C++ round-trip, the batched/GIL path, and C++→Python exception translation |

## Build & run

```bash
pip install pybind11                         # build-time dependency
python setup.py build_ext --inplace          # compiles bausatz_affine/_affine*.so
python -m pytest test_affine.py
```

Or install it as a package: `pip install --no-build-isolation .`

## What pybind11 is doing for you

- **`py::class_<AffineKernel>`** — exposes the C++ class as a Python type: constructor, methods,
  read-only properties, `__repr__`. No hand-written CPython boilerplate.
- **Type casters (`<pybind11/stl.h>`)** — `scale`/`bias` cross as plain floats; the batched method's
  `list[float]` ⇄ `std::vector<double>` conversion is automatic. This is why handing a
  pydantic-validated config into a C++ constructor needs *no* glue.
- **`py::call_guard<py::gil_scoped_release>()`** — releases the GIL around the C++ batch loop so
  other Python threads keep running; re-acquired before returning to Python.
- **Exception translation** — the kernel's `std::invalid_argument` surfaces in Python as
  `ValueError`, so native failures read like any other Python error (and sit naturally next to
  pydantic's `ValidationError`).

## Why composition (and not a trampoline)

pydantic — hence the whole config-driven machine (validation, `class_name` tag, registration,
`to_dict` round-trip) — lives in Python. Keeping it there and letting the C++ object own only the
compute is the least-friction split: the C++ side never parses or re-validates a spec.

Reach for pybind11's **trampoline** classes (`PYBIND11_OVERRIDE`) instead when the polymorphism must
cross the boundary — e.g. a C++ base whose virtual methods you want to override *in Python*, or a
C++ driver that calls back into Python-implemented stages. That's a different bridge; this example
deliberately takes the simpler one.
