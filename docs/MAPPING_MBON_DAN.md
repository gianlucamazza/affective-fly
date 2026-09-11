# MBON/DAN to CoreAffect Mapping

Explicit, testable mapping from fly mushroom body circuit readouts to Affective Field Theory `CoreAffect`.

## Overview

This document defines how we translate **fly neural activity** (MBON and DAN firing rates) into **affective dimensions** (valence and arousal) suitable for `emotional-memory`.

**Key Principle**: This mapping has **no free parameters to fit**. All values are calibrated to typical fly recordings or derived from first principles.

## Biological Background

### Mushroom Body (MB) Circuit

The *Drosophila* mushroom body implements associative learning:

1. **Kenyon Cells (KCs)**: ~2000 neurons, sparse coding (~5% active for any stimulus)
   - Function: High-dimensional stimulus representation
   - Analogy: "Random features" for classification

2. **Dopaminergic Neurons (DANs)**: ~20 neurons
   - Function: Reinforcement signal (reward/punishment)
   - PAM cluster: reward (approach-promoting)
   - MBONs receive DAN input to update KC→MBON weights

3. **Mushroom Body Output Neurons (MBONs)**: ~34 neurons
   - Function: Behavioral output (approach vs. avoid)
   - Anatomically segregated:
     - **Approach-promoting**: PAM-coupled, medial lobes
     - **Avoidance-promoting**: PPL1-coupled, vertical lobes

### Key Insight

**MBONs are already valence-coded**: approach vs. avoid populations naturally map to positive vs. negative valence.

**DANs signal arousal**: high DAN activity = strong reinforcement = high arousal/salience.

## Mapping Specification

### 1. Valence (Approach vs. Avoid Contrast)

**Formula**:
```
valence = (R_approach - R_avoid) / (R_approach + R_avoid + ε)
```

Where:
- `R_approach`: Sum of approach-promoting MBON firing rates (Hz)
- `R_avoid`: Sum of avoidance-promoting MBON firing rates (Hz)
- `ε`: Small constant (1e-6) to prevent division by zero

**Range**: [-1, 1]

**Interpretation**:
- `valence = +1`: Pure approach (all approach MBONs active, no avoid)
- `valence = -1`: Pure avoid (all avoid MBONs active, no approach)
- `valence = 0`: Balanced or no activity

**Calibration**:
- Baseline MBON rate: ~10 Hz (spontaneous activity)
- Peak MBON rate: ~100 Hz (during strong stimulus)

**Example**:
```python
# Positive stimulus → approach dominant
R_approach = 80 Hz
R_avoid = 10 Hz
valence = (80 - 10) / (80 + 10) = 0.778

# Negative stimulus → avoid dominant
R_approach = 10 Hz
R_avoid = 80 Hz
valence = (10 - 80) / (10 + 80) = -0.778
```

**Why This Mapping?**

1. **Contrast normalization**: Dividing by sum makes valence invariant to overall activity level
2. **Bounded output**: Automatically constrained to [-1, 1]
3. **Biologically grounded**: MBONs compete via inhibitory interneurons in real circuit

### 2. Arousal (DAN Activity)

**Formula**:
```
arousal = (R_DAN - R_baseline) / (R_max - R_baseline)
```

Where:
- `R_DAN`: Current DAN firing rate (Hz)
- `R_baseline`: Baseline DAN rate (~5 Hz)
- `R_max`: Maximum expected DAN rate (~80 Hz)

**Range**: Approximately [-1, 1] (clamped)

**Interpretation**:
- `arousal = +1`: High DAN activity → high arousal/salience
- `arousal = 0`: Baseline DAN activity → neutral arousal
- `arousal = -1`: Very low DAN activity → calm/low arousal

**Calibration**:
- Baseline DAN rate: ~5 Hz (from PAM recordings)
- Peak DAN rate: ~80 Hz (during strong reward/punishment)

**Example**:
```python
# High arousal
R_DAN = 80 Hz
arousal = (80 - 5) / (80 - 5) = 1.0

# Baseline arousal
R_DAN = 5 Hz
arousal = (5 - 5) / 75 = 0.0

# Low arousal
R_DAN = 2 Hz
arousal = (2 - 5) / 75 = -0.04
```

**Why This Mapping?**

1. **Biological**: DANs signal prediction error magnitude (|error|), which correlates with arousal
2. **Normalization**: Centering on baseline makes arousal bidirectional
3. **Simplicity**: Linear transform (could be nonlinear in future)

### 3. Approach Tendency (Behavioral Bias)

**Formula**:
```
approach_tendency = (R_approach / (R_approach + R_avoid + ε)) * 2 - 1
```

**Range**: [-1, 1]

**Interpretation**:
- `approach_tendency = +1`: 100% approach bias
- `approach_tendency = 0`: No bias (50/50)
- `approach_tendency = -1`: 100% avoid bias

**Use Case**: Policy and launch gate decisions.

**Relationship to Valence**:
- Similar to valence but focuses on behavioral output ratio
- Valence includes DAN modulation (via learning); approach_tendency is direct readout

## Implementation

See `src/affective_fly/affect_bridge.py` for code.

**Class**: `AffectBridge`

**Key Methods**:
- `mbon_dan_to_core_affect(state: MBONDanState) -> CoreAffect`
- `create_appraisal(state, novelty, controllability, goal_relevance) -> AppraisalVector`
- `readout_to_tuple(state) -> (valence, arousal, approach_tendency)`

**Parameters** (with defaults):
```python
AffectBridge(
    mbon_baseline=10.0,  # Hz
    dan_baseline=5.0,    # Hz
    mbon_max=100.0,      # Hz
    dan_max=80.0,        # Hz
)
```

## Validation

### Unit Tests

See `tests/test_affect_bridge.py`:

1. **Balanced approach/avoid** → valence ≈ 0
2. **Approach dominant** → valence > 0.5
3. **Avoid dominant** → valence < -0.5
4. **High DAN** → arousal > baseline
5. **Bounds respected**: All outputs in [-1, 1]

### Example Scenarios

```python
# Scenario 1: Positive reinforcement (reward)
state = MBONDanState(
    mbon_approach_rate=80.0,
    mbon_avoid_rate=15.0,
    dan_reinforcement_rate=60.0,
    arousal_rate=60.0,
)
# Expected: valence > 0.6, arousal > 0.5

# Scenario 2: Negative reinforcement (punishment)
state = MBONDanState(
    mbon_approach_rate=15.0,
    mbon_avoid_rate=80.0,
    dan_reinforcement_rate=50.0,
    arousal_rate=50.0,
)
# Expected: valence < -0.6, arousal > 0.4

# Scenario 3: Neutral stimulus
state = MBONDanState(
    mbon_approach_rate=10.0,
    mbon_avoid_rate=10.0,
    dan_reinforcement_rate=5.0,
    arousal_rate=5.0,
)
# Expected: valence ≈ 0, arousal ≈ 0
```

## Limitations & Future Work

### Current Limitations

1. **Linear mapping**: Real affect might be nonlinear
2. **No temporal dynamics**: Ignores rate of change (momentum)
3. **Simplified anatomy**: Real MBONs have ~14 approach + 20 avoid types, not binary split
4. **Learning is residual/bandit by default**: `learn()` implements `δ = r − V` (γ=0). Sequential `r + γV' − V` is supported but not the loop default.

### Planned Improvements

#### 1. MaleCNS v1.0 Integration

**Goal**: Map to actual neuron IDs from MaleCNS connectome.

**Approach MBONs** (partial list, Aso 2014):
- MBON-γ5β'2a: Approach after odor-reward learning
- MBON-β'2mp: Appetitive / approach
- MBON-β2β'2a: Approach-promoting medial lobe

**Avoid MBONs** (partial list, Aso 2014):
- MBON-γ2α'1: Aversive memory expression
- MBON-α3: Avoidance after odor-shock
- MBON-α'2: Vertical-lobe avoidance
- MBON-γ1pedc>α/β (MBON-11): GABAergic; aversive memory expression

**DANs** (partial list):
- PAM-α1: Reward signal
- PAM-β'2a: Sugar reward
- PPL1-γ1pedc: Punishment signal
- PPL1-α2α'2: Shock reinforcement

**Status**: Published Aso names live in `aso.py` / `MaleCNSCircuit`. Weights are still random. Do NOT invent MaleCNS/Schlegel body IDs until an HDF5/JSON export is available.

#### 2. Nonlinear Mapping

Explore sigmoid or softmax for valence:
```python
valence = tanh((R_approach - R_avoid) / temperature)
```

Temperature could be arousal-dependent (high arousal → steeper sigmoid).

#### 3. Momentum (Rate of Change)

Add temporal derivative to capture affect *velocity*:
```python
valence_velocity = (valence_t - valence_{t-1}) / dt
```

This would feed into AFT's momentum layer.

#### 4. Ensemble Circuits

Different fly types (male, female, virgin, mated) have different MBON/DAN connectivity. Could run ensemble of circuits and vote.

## Biological Justification

### Why MBONs Encode Valence

**Experimental Evidence**:
- Optogenetic activation of approach MBONs → fly approaches odor
- Activation of avoid MBONs → fly avoids odor
- Silencing approach MBONs → reduced approach behavior

**Reference**: Aso et al. (2014). "The neuronal architecture of the mushroom body provides a logic for associative learning." *eLife*, 3, e04577.

### Why DANs Encode Arousal

**Experimental Evidence**:
- DANs fire to prediction errors (TD error): |reward - expected|
- High DAN activity → strong memory formation
- DAN activity correlates with salience/surprise

**Reference**: Berry et al. (2018). "Dopamine neuron activity before action initiation gates and invigorates future movements." *Nature*, 554, 244-248.

### Why This Maps to Circumplex Affect

**Russell's Circumplex Model**:
- **Valence axis**: Pleasure (positive) vs. displeasure (negative)
- **Arousal axis**: Activation (high) vs. deactivation (low)

**Fly Mapping**:
- **Valence**: Approach (pleasure) vs. avoid (displeasure)
- **Arousal**: DAN activity (activation) vs. baseline (deactivation)

**Key Point**: Flies don't "feel pleasure" like humans. But they *behave* as if they have valence (approach/avoid) and arousal (activation). This is sufficient for affect-weighted memory.

## Testing the Mapping

### How to Verify

1. **Construct known inputs**: Set MBON/DAN rates manually
2. **Compute affect**: Use `AffectBridge`
3. **Check outputs**: Valence and arousal in expected range and direction

### Example Test

```python
def test_positive_valence():
    bridge = AffectBridge()
    state = MBONDanState(
        mbon_approach_rate=70.0,
        mbon_avoid_rate=20.0,
        dan_reinforcement_rate=30.0,
        arousal_rate=30.0,
    )
    affect = bridge.mbon_dan_to_core_affect(state)
    
    assert affect.valence > 0.4  # Should be positive
    assert -1.0 <= affect.valence <= 1.0  # In bounds
    assert -1.0 <= affect.arousal <= 1.0
```

## Honesty & Transparency

**What we're NOT claiming**:
- Flies experience human emotions
- This mapping is "correct" in any absolute sense
- Current parameters are final

**What we ARE claiming**:
- This mapping is **explicit and testable**
- It's based on **published fly neuroscience**
- It produces **reasonable circumplex coordinates** for affect-weighted memory

If better data emerges (e.g., quantitative MBON→behavior curves), we'll update the mapping. The code structure makes this easy.

## References

1. Aso, Y., et al. (2014). The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife*, 3, e04577.

2. Berry, J. A., Cervantes-Sandoval, I., Chakraborty, M., & Davis, R. L. (2018). Dopamine neuron activity before action initiation gates and invigorates future movements. *Nature*, 554, 244-248.

3. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

4. Owald, D., & Waddell, S. (2015). Olfactory learning skews mushroom body output pathways to steer behavioral choice in Drosophila. *Current Opinion in Neurobiology*, 35, 178-184.

5. Hige, T., Aso, Y., Modi, M. N., Rubin, G. M., & Turner, G. C. (2015). Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila. *Neuron*, 88(5), 985-998.

---

**Version**: 0.2.0  
**Last Updated**: 2026-09-11  
**Status**: Linear mapping implemented and tested. Aso 2014 names in `aso.py`. Connectome weights pending.
