# Changelog

## 0.2.6

### Renamed

- `LaunchGate` → `ActionGate`, `LaunchGateState` → `ActionGateState`, `can_launch()` → `can_act()`, `launch_gate=` / `self.launch_gate` → `action_gate`, module `launch_gate.py` → `action_gate.py`.
- Blocked-action reason: `Launch gate blocked …` → `Action gate blocked …`.
- `examples/demo_mood_launch.py` → `examples/demo_mood_gate.py` (`ACTION ALLOWED` / `ACTION BLOCKED`).
- Stimulus identity field `ticker` → `item` (key `item:{id}`).

### Deprecated

- `LaunchGate`, `LaunchGateState`, `can_launch()`, `launch_gate=` / `.launch_gate`, and `from affective_fly.launch_gate import …` remain as aliases and emit `DeprecationWarning`.
- Persisted `_stimulus_key` values with the old `ticker:{id}` prefix still match live `item:{id}` keys.
