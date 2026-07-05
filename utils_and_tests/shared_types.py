"""Shared types and the lazy-export pattern used across every module's __init__.py.

`lazy_getattr` factors out the PEP 562 `__getattr__` that every other
module in this repo hand-wrote identically (import a submodule only
when one of its names is actually accessed, so heavy/optional
dependencies like torch, brian2, or mne never load until needed). It's
provided here rather than retrofitted into the other seven modules,
since a BUILD run only touches the module it was asked for -- but any
of them could switch to `__getattr__ = lazy_getattr(_EXPORTS, __name__)`
verbatim.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

ChannelsByTime = np.ndarray
"""Conventional shape (n_channels, n_times); a type alias for documentation, not enforcement."""

EpochsChannelsTimes = np.ndarray
"""Conventional shape (n_epochs, n_channels, n_times)."""


class TaintedLike(Protocol):
    """Structural type for anything shaped like `provenance_pipeline.TaintedArray`.

    Lets code depend on "has .data and .taint" without importing
    `provenance_pipeline` directly, avoiding a cross-module import for
    what is otherwise just a type hint.
    """

    data: np.ndarray
    taint: np.ndarray


def lazy_getattr(exports: dict[str, str], module_name: str) -> Callable[[str], Any]:
    """Build a module-level `__getattr__` from a `{public_name: "pkg.submodule"}` map.

    Usage in a package's `__init__.py`::

        _EXPORTS = {"Thing": "mypackage.things"}
        __getattr__ = lazy_getattr(_EXPORTS, __name__)
    """

    def __getattr__(name: str) -> Any:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        return getattr(importlib.import_module(target), name)

    return __getattr__


if __name__ == "__main__":
    demo_exports = {"pi": "math", "sqrt": "math"}
    __getattr__ = lazy_getattr(demo_exports, "demo_module")
    print("lazy_getattr('pi') ->", __getattr__("pi"))
    print("lazy_getattr('sqrt') ->", __getattr__("sqrt"))
    try:
        __getattr__("not_a_real_export")
    except AttributeError as exc:
        print("unknown name correctly raises AttributeError:", exc)

    class HasData:
        data = np.zeros(3)
        taint = np.zeros(3, dtype=np.uint8)

    obj: TaintedLike = HasData()
    print(f"TaintedLike accepts a duck-typed object: data.shape={obj.data.shape}")
    print("OK: lazy_getattr factory and TaintedLike protocol both work as intended.")
