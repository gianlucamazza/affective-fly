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
        n_kc: Number of Kenyon cells. Must equal the connectome KC count
            when ``connectivity_path`` is set, unless ``allow_kc_mismatch``.
        seed: Random seed (used only when connectivity_path is None)
        connectivity_path: Path to connectome file (JSON/Feather/Parquet).
            When None, weights are random (NOT connectome-backed).
            When provided, KC→MBON weights are loaded from the file.
        allow_kc_mismatch: If False (default), raise when the file's KC count
            differs from ``n_kc``. If True, truncate extra KCs or pad with zeros.
        syn_gain: Override LIF ``syn_gain``. ``None`` (default) refits from
            KC→MBON fan-in after a connectome load. Ignored by Brian2
            (band held by ``w_max``; pass ``syn_gain`` on ``Brian2Circuit``).
        codegen_target: Passed to ``Brian2Circuit`` when ``backend="brian2"``.
            ``numpy`` (default) needs no compiler; ``cpp_standalone`` is opt-in.
        build_dir: Standalone build/cache directory for the Brian2 C++ path.
    """

    def __init__(
        self,
        backend: Literal["lif", "brian2"] | FlyAffectReadout = "lif",
        catalog: AsoCatalog | None = None,
        n_kc: int = 200,
        seed: int = 42,
        connectivity_path: str | Path | None = None,
        allow_kc_mismatch: bool = False,
        syn_gain: float | None = None,
        codegen_target: str = "numpy",
        build_dir: str | Path | None = None,
    ):
        self.catalog = catalog or ASO_CATALOG
        self.mbon_names = self.catalog.mbon_names
        self.dan_names = self.catalog.dan_names
        self.last_state: MBONDanState | None = None
        self.connectivity_path = connectivity_path
        self.connectivity_loaded = False
        self.matched_mbon_count = 0  # Number of catalog MBONs that received weights
        self.matched_mbon_names: list[str] = []
        self.allow_kc_mismatch = allow_kc_mismatch
        self.kc_rows_in_file: int | None = None

        if isinstance(backend, FlyAffectReadout):
            self.backend = backend
            if syn_gain is not None:
                self._apply_syn_gain_override(syn_gain)
        elif backend == "lif":
            self.backend = LIFCircuit(
                n_kc=n_kc,
                n_dan=self.catalog.n_dan,
                n_mbon=self.catalog.n_mbon,
                n_approach=self.catalog.n_approach,
                n_pam=self.catalog.n_pam,
                seed=seed,
                syn_gain=syn_gain,
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
                codegen_target=codegen_target,
                build_dir=build_dir,
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
        """Population-rate aliases keyed by Aso cell-type name.

        This is **not** a per-cell spike rate. The backend exposes three
        population rates only: ``mbon_approach_rate``, ``mbon_avoid_rate``,
        and ``dan_reinforcement_rate``. Every approach MBON name receives the
        same approach population rate; every avoid name receives the avoid
        rate; every DAN name receives the DAN rate.

        Use :meth:`mbon_index_rates` / :meth:`dan_index_rates` for the same
        aliases ordered by catalog index.
        """
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

    def mbon_index_rates(self) -> list[float]:
        """Population-rate alias for each catalog MBON index (not per-cell)."""
        if self.last_state is None:
            return [0.0] * len(self.mbon_names)
        n_app = self.catalog.n_approach
        app = self.last_state.mbon_approach_rate
        avoid = self.last_state.mbon_avoid_rate
        return [app if i < n_app else avoid for i in range(len(self.mbon_names))]

    def dan_index_rates(self) -> list[float]:
        """Population-rate alias for each catalog DAN index (not per-cell)."""
        if self.last_state is None:
            return [0.0] * len(self.dan_names)
        rate = self.last_state.dan_reinforcement_rate
        return [rate] * len(self.dan_names)

    def _load_connectivity(self) -> None:
        """Load KC→MBON weights from connectivity_path.

        Updates backend.w_kc_mbon with loaded weights.
        Raises ConnectomeLoadError if file is missing/unreadable or if the
        file KC count does not match the backend and allow_kc_mismatch is False.
        """
        from .malecns_connectome import ConnectomeLoadError, load_connectome, map_to_aso_names

        if self.connectivity_path is None:
            return

        connectivity = load_connectome(self.connectivity_path)
        mapped_weights, matched = map_to_aso_names(connectivity, list(self.mbon_names))

        self.kc_rows_in_file = int(mapped_weights.shape[0])
        self.matched_mbon_count = len(matched)
        self.matched_mbon_names = list(matched)

        # Warn if mapping matched few or no catalog MBONs
        if self.matched_mbon_count == 0:
            import warnings

            warnings.warn(
                f"map_to_aso_names matched 0 MBONs from {len(connectivity.mbon_body_ids)} loaded. "
                f"Weights will be all zeros. Check MaleCNS type / Aso catalog coverage.",
                UserWarning,
                stacklevel=2,
            )
        elif self.matched_mbon_count < len(self.mbon_names) // 2:
            import warnings

            warnings.warn(
                f"map_to_aso_names matched only {self.matched_mbon_count}/{len(self.mbon_names)} "
                f"MBONs. Coverage is partial. Unmatched MBONs: "
                f"{set(self.mbon_names) - set(matched)}",
                UserWarning,
                stacklevel=2,
            )

        # Update backend weights if it has w_kc_mbon
        if hasattr(self.backend, "w_kc_mbon"):
            # Ensure shapes match
            if mapped_weights.shape[1] != self.backend.w_kc_mbon.shape[1]:
                raise ValueError(
                    f"Loaded connectivity has {mapped_weights.shape[1]} MBONs, "
                    f"but backend expects {self.backend.w_kc_mbon.shape[1]}"
                )
            n_kc_backend = self.backend.w_kc_mbon.shape[0]
            if mapped_weights.shape[0] != n_kc_backend:
                if not self.allow_kc_mismatch:
                    raise ConnectomeLoadError(
                        f"Connectome has {mapped_weights.shape[0]} KCs but backend "
                        f"n_kc={n_kc_backend}. Pass n_kc={mapped_weights.shape[0]} "
                        f"to match the file, or allow_kc_mismatch=True to "
                        f"truncate extra rows / pad missing rows with zeros."
                    )
                if mapped_weights.shape[0] > n_kc_backend:
                    mapped_weights = mapped_weights[:n_kc_backend, :]
                else:
                    pad_rows = n_kc_backend - mapped_weights.shape[0]
                    mapped_weights = np.vstack(
                        [mapped_weights, np.zeros((pad_rows, mapped_weights.shape[1]))]
                    )
            self.backend.w_kc_mbon = mapped_weights
            self.connectivity_loaded = True
            refresh = getattr(self.backend, "refresh_default_syn_gain", None)
            if callable(refresh):
                refresh()

    def _apply_syn_gain_override(self, syn_gain: float) -> None:
        """Force a caller-supplied syn_gain on a prebuilt LIF backend."""
        backend = self.backend
        if hasattr(backend, "syn_gain"):
            setattr(backend, "syn_gain", syn_gain)
        if hasattr(backend, "dan_syn_gain"):
            setattr(backend, "dan_syn_gain", syn_gain)
        if hasattr(backend, "_syn_gain_overridden"):
            setattr(backend, "_syn_gain_overridden", True)
        if hasattr(backend, "_syn_gain_override"):
            setattr(backend, "_syn_gain_override", syn_gain)
