"""
Published mushroom-body cell types (Aso et al. 2014).

These are literature names, not MaleCNS/Schlegel body IDs and not a
connectivity matrix. ``MaleCNSCircuit`` weights stay random unless
``connectivity_path`` is provided. Do not invent numeric IDs here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NamedCell:
    """A published MBON or DAN type with a valence class."""

    name: str
    valence: str  # approach | avoid | pam | ppl1
    note: str
    source: str = "Aso et al. 2014 eLife e04577"


# Polarities from Aso 2014 / Owald & Waddell 2015: medial-lobe MBONs tend to
# promote approach; vertical-lobe MBONs tend to promote avoidance.
# MBON-gamma1pedc>alpha/beta (MBON-11) expresses aversive memory (avoid).
APPROACH_MBONS: tuple[NamedCell, ...] = (
    NamedCell("MBON-gamma5beta'2a", "approach", "Appetitive memory expression"),
    NamedCell("MBON-beta'2mp", "approach", "Appetitive / approach"),
    NamedCell("MBON-beta2beta'2a", "approach", "Approach-promoting medial lobe"),
)

AVOID_MBONS: tuple[NamedCell, ...] = (
    NamedCell("MBON-gamma2alpha'1", "avoid", "Aversive memory expression"),
    NamedCell("MBON-alpha3", "avoid", "Avoidance after odor-shock"),
    NamedCell("MBON-alpha'2", "avoid", "Vertical-lobe avoidance"),
    NamedCell(
        "MBON-gamma1pedc>alpha/beta",
        "avoid",
        "GABAergic MBON-11; aversive memory expression",
    ),
)

PAM_DANS: tuple[NamedCell, ...] = (
    NamedCell("PAM-alpha1", "pam", "Reward / sugar-related PAM"),
    NamedCell("PAM-beta'2a", "pam", "Sugar reward PAM"),
)

PPL1_DANS: tuple[NamedCell, ...] = (
    NamedCell("PPL1-gamma1pedc", "ppl1", "Punishment; also MB-MP1"),
    NamedCell("PPL1-alpha2alpha'2", "ppl1", "Shock reinforcement"),
)


@dataclass(frozen=True)
class AsoCatalog:
    approach_mbons: tuple[NamedCell, ...] = APPROACH_MBONS
    avoid_mbons: tuple[NamedCell, ...] = AVOID_MBONS
    pam_dans: tuple[NamedCell, ...] = PAM_DANS
    ppl1_dans: tuple[NamedCell, ...] = PPL1_DANS

    @property
    def mbon_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.approach_mbons + self.avoid_mbons)

    @property
    def dan_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.pam_dans + self.ppl1_dans)

    @property
    def n_approach(self) -> int:
        return len(self.approach_mbons)

    @property
    def n_avoid(self) -> int:
        return len(self.avoid_mbons)

    @property
    def n_mbon(self) -> int:
        return self.n_approach + self.n_avoid

    @property
    def n_pam(self) -> int:
        return len(self.pam_dans)

    @property
    def n_ppl1(self) -> int:
        return len(self.ppl1_dans)

    @property
    def n_dan(self) -> int:
        return self.n_pam + self.n_ppl1


ASO_CATALOG = AsoCatalog()
