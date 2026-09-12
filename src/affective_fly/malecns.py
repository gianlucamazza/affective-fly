"""
Named MB circuit using published Aso types.

Supports loading real KC→MBON weights from published MaleCNS connectivity
(via connectivity_path parameter). When connectivity_path is omitted, falls
back to seeded-random weights.
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
        backend: Circuit backend ("lif", "brian2", or FlyAffectReadout instance)
        catalog: Aso catalog (defaults to ASO_CATALOG)
        n_kc: Number of Kenyon Cells
        seed: Random seed
        connectivity_path: Optional path to real KC→MBON connectivity file
            (feather/parquet/json/csv from scripts/filter_malecns_connectivity.py).
            When provided, loads REAL published weights. When omitted, falls back
            to random initialization.
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
        self.connectivity_path = Path(connectivity_path) if connectivity_path else None

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

        # Load real connectivity if path provided
        if self.connectivity_path is not None:
            self._load_real_connectivity(seed)

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

    def _load_real_connectivity(self, seed: int) -> None:
        """Load real KC→MBON weights from connectivity_path into backend.
        
        Only supports LIFCircuit and Brian2Circuit backends for now.
        Raises a warning if backend doesn't support weight loading.
        """
        if self.connectivity_path is None or not self.connectivity_path.exists():
            print(f"Warning: connectivity_path {self.connectivity_path} not found, using random weights")
            return
        
        # Check if backend has n_kc attribute
        if not hasattr(self.backend, 'n_kc'):
            print(f"Warning: backend {type(self.backend).__name__} does not expose n_kc, cannot load weights")
            return
        
        from .malecns_connectome import load_connectivity, map_connectivity_to_circuit
        
        print(f"Loading real KC→MBON connectivity from {self.connectivity_path}")
        conn = load_connectivity(self.connectivity_path)
        
        # Map to circuit weight matrix
        w_kc_mbon = map_connectivity_to_circuit(
            conn,
            n_kc=self.backend.n_kc,  # type: ignore[attr-defined]
            n_mbon=self.catalog.n_mbon,
            catalog_mbon_names=list(self.mbon_names),
            seed=seed,
            w_max=getattr(self.backend, 'w_max', 0.15),
        )
        
        # Install weights into backend
        if isinstance(self.backend, LIFCircuit):
            self.backend.w_kc_mbon = w_kc_mbon
            print("Loaded real weights into LIFCircuit.w_kc_mbon")
        elif hasattr(self.backend, 'w_kc_mbon'):
            # Brian2Circuit or similar
            self.backend.w_kc_mbon = w_kc_mbon  # type: ignore[attr-defined]
            print("Loaded real weights into backend.w_kc_mbon")
        else:
            print(f"Warning: backend {type(self.backend).__name__} does not expose w_kc_mbon, cannot load weights")
