"""Deprecated aliases for :class:`ActionGate`.

Import ``ActionGate`` / ``ActionGateState`` from ``affective_fly`` instead.
This module exists so ``from affective_fly.launch_gate import LaunchGate``
keeps working until callers migrate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from .action_gate import ActionGate, ActionGateState


@dataclass
class LaunchGateState(ActionGateState):
    """Deprecated alias of :class:`ActionGateState`."""

    def __post_init__(self) -> None:
        warnings.warn(
            "LaunchGateState is deprecated; use ActionGateState",
            DeprecationWarning,
            stacklevel=2,
        )


class LaunchGate(ActionGate):
    """Deprecated alias of :class:`ActionGate`."""

    def __init__(
        self,
        threshold_approach: float = 0.2,
        threshold_valence: float = -0.1,
        required_ticks: int = 3,
    ):
        warnings.warn(
            "LaunchGate is deprecated; use ActionGate",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            threshold_approach=threshold_approach,
            threshold_valence=threshold_valence,
            required_ticks=required_ticks,
        )


def _deprecated_can_launch(self: ActionGate) -> bool:
    warnings.warn(
        "can_launch() is deprecated; use can_act()",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.can_act()


setattr(ActionGate, "can_launch", _deprecated_can_launch)
