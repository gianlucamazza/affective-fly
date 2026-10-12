"""NumPy 2 compatibility shims.

NumPy 2.4 removed ``ndarray.ptp`` (use ``np.ptp``). Brian2 2.9 still binds
``Quantity.ptp = wrap_function_keep_dimensions(np.ndarray.ptp)`` at import
time, which raises ``AttributeError`` before any circuit code runs. Brian2
2.10 has the upstream fix but requires Python 3.12; this project stays on
3.11, so we rewrite that one binding while the module loads.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

_PTP_OLD = "ptp = wrap_function_keep_dimensions(np.ndarray.ptp)"
_PTP_NEW = "ptp = wrap_function_keep_dimensions(np.ptp)"
_TARGET = "brian2.units.fundamentalunits"


class _Brian2PtpLoader(Loader):
    def __init__(self, origin: str) -> None:
        self.origin = origin

    def create_module(self, spec: ModuleSpec) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        source = Path(self.origin).read_text(encoding="utf-8")
        source = source.replace(_PTP_OLD, _PTP_NEW)
        exec(compile(source, self.origin, "exec"), module.__dict__)


class _Brian2PtpFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: Any = None,
    ) -> ModuleSpec | None:
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.origin is None:
                continue
            spec.loader = _Brian2PtpLoader(spec.origin)
            return spec
        return None


def install_numpy2_compat() -> None:
    """Install the Brian2 ``ndarray.ptp`` import hook if NumPy 2.4+ is in use."""
    import numpy as np

    if hasattr(np.ndarray, "ptp"):
        return
    if any(isinstance(finder, _Brian2PtpFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _Brian2PtpFinder())


def array_copy(values: Any, dtype: Any) -> Any:
    """Copy ``values`` to a new ndarray (NumPy 2 ``copy=True`` / dtype rules).

    ``np.array(..., copy=False)`` now raises when a copy is required. Callers
    that need a snapshot (Brian2 monitor counts, membrane voltages) must
    request the copy explicitly.
    """
    import numpy as np

    return np.array(values, dtype=dtype, copy=True)


install_numpy2_compat()
