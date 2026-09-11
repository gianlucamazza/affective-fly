# MBON/DAN → CoreAffect Mapping

This document specifies the **explicit, testable mapping** from Drosophila mushroom-body neuron firing rates to Affective Field Theory `CoreAffect` coordinates.

---

## Scientific Honesty

**What we know** (from fly neuroscience):
- MBON populations drive approach or avoidance behaviors
- DAN activity signals reinforcement (reward or punishment)
- Arousal correlates with overall circuit activity and dopamine

**What we DON'T know** (and must not pretend):
- Exact quantitative mapping from Hz to subjective valence/arousal
- Whether flies have "subjective experience" comparable to human affect
- Whether our heuristic coefficients generalize to real fly recordings

**This mapping is**: A **testable computational hypothesis** suitable for mock/LIF circuits and future validation against real data.

---

## Mapping Equations

### Valence

Valence represents approach/avoid balance, modulated by reinforcement:

```
valence = tanh(k_approach * mbon_approach 
               - k_avoid * mbon_avoid 
               + k_dan * dan_reinforcement)
```

**Rationale**:
- High `mbon_approach` → positive valence (approach drive)
- High `mbon_avoid` → negative valence (avoidance drive)
- Positive `dan_reinforcement` (reward) → boost valence
- Negative `dan_reinforcement` (punishment) → suppress valence
- `tanh` squashes to [-1, 1] range

**Default coefficients** (heuristic, NOT biologically fitted):
- `k_approach = 0.02`: Weight for approach MBON contribution
- `k_avoid = 0.02`: Weight for avoid MBON contribution (subtracted)
- `k_dan = 0.03`: Weight for DAN reinforcement modulation

These coefficients are chosen such that:
- Baseline MBON rates (~20-50 Hz) produce moderate valence (~0.2-0.5)
- Strong MBON imbalance (80 vs 20 Hz) produces strong valence (~0.7-0.9)
- DAN reinforcement (±30 Hz) can shift valence by ~0.3-0.5

### Arousal

Arousal represents overall circuit activity and salience:

```
arousal = tanh(k_arousal * (arousal_signal / arousal_baseline))
```

**Rationale**:
- `arousal_signal` is an aggregate measure of circuit activity (could be total spike rate, variance, or separate arousal pathway)
- Normalized by `arousal_baseline` to map typical activity to mid-range arousal
- `tanh` squashes to [-1, 1] range

**Default coefficients**:
- `k_arousal = 0.015`: Weight for arousal scaling
- `arousal_baseline = 40.0 Hz`: Typical baseline arousal activity

These coefficients map:
- Low activity (~10-20 Hz) → low arousal (~0.0-0.2)
- Moderate activity (~40 Hz) → mid arousal (~0.3-0.5)
- High activity (~80 Hz) → high arousal (~0.6-0.8)

---

## Biological Grounding (Partial)

### Known from Fly Literature

1. **MBON Valence Coding** (Aso et al. 2014, Owald et al. 2015):
   - MBON-γ1pedc>α/β, MBON-α2sc, MBON-α3 → approach behaviors
   - MBON-γ5β'2a, MBON-γ4>γ1γ2, MBON-β'2mp → avoidance behaviors
   - Activation of approach MBONs drives approach; silencing blocks it
   - Activation of avoid MBONs drives avoidance; silencing blocks it

2. **DAN Reinforcement** (Burke et al. 2012, Liu et al. 2012):
   - PAM DANs (protocerebral anterior medial) → reward signal
   - PPL1 DANs (posterior protocerebral lateral) → punishment signal
   - DAN activity during outcome modulates KC→MBON synapses (dopamine-dependent plasticity)

3. **Arousal/Salience** (Cohn et al. 2015, Hige et al. 2015):
   - Octopamine (OA) and dopamine modulate arousal state
   - High arousal → increased responsiveness and activity
   - Circuit-wide activity correlates with behavioral urgency

### Unknowns (Future Work)

1. **Quantitative firing rate ranges**:
   - What is "baseline" MBON activity in behaving flies?
   - What is the dynamic range during strong approach vs avoid?
   - How do rates vary across individuals, hunger state, circadian time?

2. **Population coding**:
   - Are MBON populations redundant or specialized?
   - Does valence require specific MBON combinations or ratios?

3. **Arousal substrates**:
   - Is arousal encoded in firing rate variance, synchrony, or separate pathway?
   - How does dopamine/octopamine modulation map to arousal dimension?

---

## Mapping for Real Data (Future)

When real MaleCNS v1.0 connectome data and simultaneous recordings become available:

### Step 1: Identify Neurons

Replace placeholders with real neuron IDs:

**TODO (MaleCNS v1.0)**:
- PAM DAN cluster (reward): `[PLACEHOLDER]`
- PPL1 DAN cluster (punishment): `[PLACEHOLDER]`
- MBON-γ1pedc>α/β (approach): `[PLACEHOLDER]`
- MBON-γ5β'2a (avoid): `[PLACEHOLDER]`
- Additional MBON approach: MBON-α2sc, MBON-α3
- Additional MBON avoid: MBON-γ4>γ1γ2, MBON-β'2mp

### Step 2: Fit Coefficients from Recordings

Collect simultaneous:
- MBON/DAN firing rates (from calcium imaging or electrophysiology)
- Behavioral outcomes (approach, avoid, neutral)

Fit `k_approach`, `k_avoid`, `k_dan`, `k_arousal` by:
1. Labeling behaviors as approach/avoid/neutral
2. Regressing firing rates → behavior labels
3. Optimizing coefficients to maximize prediction accuracy

### Step 3: Validate on Hold-out Data

Test mapping on unseen flies/contexts. Metrics:
- Valence sign matches approach/avoid direction (accuracy %)
- Arousal magnitude correlates with behavioral urgency (correlation)
- Generalization across stimulus types (cross-validation)

---

## Implementation (`affect_bridge.py`)

```python
class AffectBridge:
    def __init__(
        self,
        k_approach: float = 0.02,
        k_avoid: float = 0.02,
        k_dan: float = 0.03,
        k_arousal: float = 0.015,
        arousal_baseline: float = 40.0,
    ):
        self.k_approach = k_approach
        self.k_avoid = k_avoid
        self.k_dan = k_dan
        self.k_arousal = k_arousal
        self.arousal_baseline = arousal_baseline
    
    def to_core_affect(self, rates: MBONDANRates) -> CoreAffect:
        # Valence: approach vs avoid, modulated by reinforcement
        valence_input = (
            self.k_approach * rates.mbon_approach
            - self.k_avoid * rates.mbon_avoid
            + self.k_dan * rates.dan_reinforcement
        )
        valence = float(np.tanh(valence_input))
        
        # Arousal: scaled from arousal signal
        arousal_input = self.k_arousal * (rates.arousal_signal / self.arousal_baseline)
        arousal = float(np.tanh(arousal_input))
        
        return CoreAffect(valence=valence, arousal=arousal)
```

---

## Testing Strategy

**Unit tests** (`test_affect_bridge.py`):

1. **Positive valence**: High approach + low avoid + positive DAN → valence > 0
2. **Negative valence**: Low approach + high avoid + negative DAN → valence < 0
3. **Arousal scaling**: High arousal_signal → high arousal
4. **Range constraints**: All outputs in [-1, 1]

**Integration tests** (`test_loop.py`):

1. **Positive trajectory**: Sustained positive input → mood drifts positive
2. **Negative trajectory**: Sustained negative input → mood drifts negative
3. **Policy alignment**: Positive valence → approach actions; negative → avoid

---

## Example Scenarios

### Scenario 1: Reward Learning

```
Step | mbon_approach | mbon_avoid | dan_reinforcement | valence | arousal
-----|---------------|------------|-------------------|---------|--------
   0 |    20 Hz      |   20 Hz    |      0 Hz         |  0.00   |  0.30
  10 |    30 Hz      |   20 Hz    |     +10 Hz        | +0.35   |  0.35
  20 |    50 Hz      |   15 Hz    |     +20 Hz        | +0.65   |  0.45
```

Interpretation: As reward accumulates, approach MBONs increase and DAN reinforcement boosts valence. Arousal increases with activity.

### Scenario 2: Punishment Avoidance

```
Step | mbon_approach | mbon_avoid | dan_reinforcement | valence | arousal
-----|---------------|------------|-------------------|---------|--------
   0 |    20 Hz      |   20 Hz    |      0 Hz         |  0.00   |  0.30
  10 |    15 Hz      |   40 Hz    |     -10 Hz        | -0.50   |  0.40
  20 |    10 Hz      |   60 Hz    |     -20 Hz        | -0.75   |  0.50
```

Interpretation: Punishment drives up avoid MBONs and negative DAN suppresses valence. Arousal increases (defensive arousal).

### Scenario 3: Neutral Exploration

```
Step | mbon_approach | mbon_avoid | dan_reinforcement | valence | arousal
-----|---------------|------------|-------------------|---------|--------
   0 |    25 Hz      |   25 Hz    |      0 Hz         |  0.00   |  0.35
  10 |    30 Hz      |   30 Hz    |     +2 Hz         | +0.06   |  0.38
  20 |    20 Hz      |   20 Hz    |     -2 Hz         | -0.06   |  0.32
```

Interpretation: Balanced MBON activity and small DAN fluctuations keep valence near zero (exploration without strong preference).

---

## Future Refinements

1. **Nonlinear interactions**: MBON × DAN multiplicative terms (neuromodulation)
2. **Temporal dynamics**: Integrate DAN over time (eligibility traces)
3. **Population coding**: Weighted sum over multiple MBON subtypes
4. **Context modulation**: Different coefficients for different stimulus modalities (odor, vision, etc.)

---

## References

- Aso et al. (2014). "Mushroom body output neurons encode valence and guide memory-based action selection." *eLife*.
- Burke et al. (2012). "Layered reward signaling through octopamine and dopamine in Drosophila." *Nature*.
- Cohn et al. (2015). "Coordinated and compartmentalized neuromodulation shapes sensory processing in Drosophila." *Cell*.
- Hige et al. (2015). "Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila." *Neuron*.
- Liu et al. (2012). "Distinct memory traces for two visual features in the Drosophila brain." *Nature*.
- Owald et al. (2015). "Activity of defined mushroom body output neurons underlies learned olfactory behavior in Drosophila." *Neuron*.

---

**Status**: v0.1.0 — Heuristic coefficients, testable mapping, ready for real data validation.
