"""
Named MB circuit using published Aso types.

When connectivity_path is provided, KC→MBON weights are loaded from a published
connectome export (JSON/Feather/Parquet). Without a path, weights remain seeded-random.

**Important**: Random weights are NOT connectome-backed. They are placeholders
until real connectivity data is provided via connectivity_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .aso import ASO_CATALOG, AsoCatalog
from .fly_circuit import FlyAffectReadout, LIFCircuit, MBONDanState

if TYPE_CHECKING:
    from .td import TDResult


class MaleCNSCircuit(FlyAffectReadout):
    """FlyAffectReadout whose MBON/DAN indices map to published Aso names.

    Args:
        backend: "lif", "brian2", or a FlyAffectReadout instance
        catalog: Aso catalog (defaults to ASO_CATALOG)
        n_kc: Number of Kenyon cells
        seed: Random seed (used only when connectivity_path is None)
        connectivity_path: Path to connectome file (JSON/Feather/Parquet).
            When None, weights are random (NOT connectome-backed).
            When provided, KC→MBON weights are loaded from the file.
    """

    def __init__(
        self,
        backend: Literal["lif", "brian2"] | FlyAffectReadout = "lif",
        catalog: AsoCatalog | None = None,
        n_kc: int = 200,
        seed: int = 42,
        connectivity_path: str | Path | None = None,
    ):
        self.catalog = catalog or ASO_CATALOG
        self.mbon_names = self.catalog.mbon_names
        self.dan_names = self.catalog.dan_names
        self.last_state: MBONDanState | None = None
        self.connectivity_path = connectivity_path
        self.connectivity_loaded = False

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

        # Load connectome if path provided
        if connectivity_path is not None:
            self._load_connectivity()

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
    ) -> TDResult | None:
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

    def _load_connectivity(self) -> None:
        """Load KC→MBON weights from connectivity_path.

        Updates backend.w_kc_mbon with loaded weights.
        Raises ConnectomeLoadError if file is missing/unreadable.
        """
        from .malecns_connectome import load_connectome, map_to_aso_names

        if self.connectivity_path is None:
            return

        connectivity = load_connectome(self.connectivity_path)
        mapped_weights, matched = map_to_aso_names(connectivity, list(self.mbon_names))

        # Update backend weights if it has w_kc_mbon
        if hasattr(self.backend, "w_kc_mbon"):
            # Ensure shapes match
            if mapped_weights.shape[1] != self.backend.w_kc_mbon.shape[1]:
                raise ValueError(
                    f"Loaded connectivity has {mapped_weights.shape[1]} MBONs, "
                    f"but backend expects {self.backend.w_kc_mbon.shape[1]}"
                )
            # Resize if KC count differs
            if mapped_weights.shape[0] != self.backend.w_kc_mbon.shape[0]:
                n_kc_backend = self.backend.w_kc_mbon.shape[0]
                if mapped_weights.shape[0] > n_kc_backend:
                    # Truncate
                    mapped_weights = mapped_weights[:n_kc_backend, :]
                else:
                    # Pad with zeros
                    pad_rows = n_kc_backend - mapped_weights.shape[0]
                    mapped_weights = np.vstack(
                        [mapped_weights, np.zeros((pad_rows, mapped_weights.shape[1]))]
                    )
            self.backend.w_kc_mbon = mapped_weights
            self.connectivity_loaded = True
