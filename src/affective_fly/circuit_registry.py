"""Pick a circuit backend by name.

Hosts and the live runner call ``get_circuit("lif")`` (or mock / brian2 /
malecns) instead of scattering constructor ifs. Constructor kwargs are
forwarded unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .fly_circuit import FlyAffectReadout, LIFCircuit, MockFlyCircuit

CIRCUIT_NAMES = ("mock", "lif", "brian2", "malecns")


def _brian2_circuit(**kwargs: Any) -> FlyAffectReadout:
    from .brian2_circuit import Brian2Circuit

    return Brian2Circuit(**kwargs)


def _malecns_circuit(**kwargs: Any) -> FlyAffectReadout:
    from .malecns import MaleCNSCircuit

    return MaleCNSCircuit(**kwargs)


_BUILDERS: dict[str, Callable[..., FlyAffectReadout]] = {
    "mock": MockFlyCircuit,
    "lif": LIFCircuit,
    "brian2": _brian2_circuit,
    "malecns": _malecns_circuit,
}


def get_circuit(name: str, **kwargs: Any) -> FlyAffectReadout:
    """Construct a ``FlyAffectReadout`` by registry name.

    Args:
        name: ``mock``, ``lif``, ``brian2``, or ``malecns`` (case-insensitive).
        **kwargs: Forwarded to the backend constructor.

    Raises:
        ValueError: Unknown name.
        ImportError: ``brian2`` requested but the extra is not installed.
    """
    key = name.strip().lower()
    builder = _BUILDERS.get(key)
    if builder is None:
        known = ", ".join(CIRCUIT_NAMES)
        raise ValueError(f"Unknown circuit {name!r}. Choose one of: {known}.")
    return builder(**kwargs)
