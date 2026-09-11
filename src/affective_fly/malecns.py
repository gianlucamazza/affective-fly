"""
Named MB circuit using published Aso types.

This is NOT a MaleCNS connectome loader. It sizes KC/DAN/MBON populations
to the Aso catalog and labels them. Synaptic weights remain seeded-random
until an HDF5/JSON export is available.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .aso import ASO_CATALOG, AsoCatalog
from .fly_circuit import FlyAffectReadout, LIFCircuit, MBONDanState


class MaleCNSCircuit(FlyAffectReadout):
    """FlyAffectReadout whose MBON/DAN indices map to published Aso names."""

    def __init__(
        self,
        backend: Literal["lif", "brian2"] | FlyAffectReadout = "lif",
        catalog: AsoCatalog | None = None,
        n_kc: int = 200,
        seed: int = 42,
    ):
        self.catalog = catalog or ASO_CATALOG
        self.mbon_names = self.catalog.mbon_names
        self.dan_names = self.catalog.dan_names
        self.last_state: MBONDanState | None = None

        if isinstance(backend, FlyAffectReadout):
            self.backend = backend
        elif backend == "lif":
            self.backend = LIFCircuit(
                n_kc=n_kc,
                n_dan=self.catalog.n_dan,
                n_mbon=self.catalog.n_mbon,
                n_approach=self.catalog.n_approach,
                n_pam=self.catalog.n_pam,
                seed=seed,
            )
        elif backend == "brian2":
            from .brian2_circuit import Brian2Circuit

            self.backend = Brian2Circuit(
                n_kc=n_kc,
                n_dan=self.catalog.n_dan,
                n_mbon=self.catalog.n_mbon,
                n_approach=self.catalog.n_approach,
                n_pam=self.catalog.n_pam,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def step(self, sensory_input: np.ndarray, dt: float = 0.001) -> MBONDanState:
        self.last_state = self.backend.step(sensory_input, dt=dt)
        return self.last_state

    def reset(self) -> None:
        self.backend.reset()
        self.last_state = None

    def learn(
        self,
        reward: float,
        *,
        v_next: float | None = None,
        sequential: bool = False,
        prediction_error: bool | None = None,
    ):
        return self.backend.learn(
            reward,
            v_next=v_next,
            sequential=sequential,
            prediction_error=prediction_error,
        )

    def named_rates(self) -> dict[str, float]:
        """Last-step rates keyed by Aso cell-type name."""
        if self.last_state is None:
            return {name: 0.0 for name in self.mbon_names + self.dan_names}
        out: dict[str, float] = {}
        n_app = self.catalog.n_approach
        for i, name in enumerate(self.mbon_names):
            out[name] = (
                self.last_state.mbon_approach_rate if i < n_app else self.last_state.mbon_avoid_rate
            )
        for name in self.dan_names:
            out[name] = self.last_state.dan_reinforcement_rate
        return out
