"""Deprecated names and persisted-key compatibility. Not public API.

Kept so old imports and on-disk ``_stimulus_key`` values still resolve.
Do not use these identifiers in new code.
"""

from __future__ import annotations

# Historical stimulus-identity field / key prefix (replaced by ``item`` / ``item:``).
LEGACY_ITEM_FIELD = "ticker"
LEGACY_ITEM_KEY_PREFIX = "ticker:"

# Historical blocked-action reason substring (replaced by ``action gate blocked``).
LEGACY_GATE_BLOCKED_MARKER = "launch gate blocked"

DEPRECATED_EXPORTS = ("LaunchGate", "LaunchGateState")
REMOVAL_NOTE = "will be removed in 0.3.0"
