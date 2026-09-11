"""Bridge MBON/DAN firing rates to emotional-memory CoreAffect coordinates.

This module implements the mapping from fly mushroom-body circuit readout
(MBON approach/avoid + DAN reinforcement + arousal) to the valence-arousal
circumplex space used by Affective Field Theory.

Mapping is explicit, testable, and documented. NO fake training or invented
connectome IDs. Placeholders for MaleCNS v1.0 neuron IDs are clearly marked.
"""

from dataclasses import dataclass

import numpy as np

from .fly_circuit import MBONDANRates

try:
    from emotional_memory.core import AppraisalVector, CoreAffect
except ImportError:
    # Fallback for type checking when emotional-memory not installed
    @dataclass
    class CoreAffect:
        valence: float
        arousal: float

    @dataclass
    class AppraisalVector:
        valence: float
        arousal: float
        dominance: float = 0.5
        unpredictability: float = 0.5


class AffectBridge:
    """Map MBON/DAN rates to CoreAffect (valence, arousal) in [-1, 1].
    
    Mapping strategy (see docs/MAPPING_MBON_DAN.md for full details):
    
    1. **Valence**: Computed from the balance of approach vs avoid MBONs,
       modulated by DAN reinforcement signal.
       
       valence = tanh(k_approach * mbon_approach - k_avoid * mbon_avoid 
                      + k_dan * dan_reinforcement)
       
    2. **Arousal**: Derived from total circuit activity (arousal_signal),
       scaled and clipped to [-1, 1].
       
       arousal = tanh(k_arousal * arousal_signal / baseline)
    
    3. **Approach/Avoid**: Direct readout for policy decisions.
       
       approach_drive = mbon_approach - mbon_avoid
       
    Default coefficients are heuristic estimates suitable for mock/LIF circuits.
    For real MaleCNS connectome data, coefficients should be fitted from recordings.
    
    TODO(MaleCNS v1.0): Replace with real neuron IDs when connectome available:
      - PAM DAN cluster (reward): PLACEHOLDER
      - PPL1 DAN cluster (punishment): PLACEHOLDER  
      - MBON-γ1pedc>α/β (approach): PLACEHOLDER
      - MBON-γ5β'2a (avoid): PLACEHOLDER
    """
    
    def __init__(
        self,
        k_approach: float = 0.02,
        k_avoid: float = 0.02,
        k_dan: float = 0.03,
        k_arousal: float = 0.015,
        arousal_baseline: float = 40.0,
    ):
        """Initialize affect bridge with mapping coefficients.
        
        Args:
            k_approach: Weight for MBON approach contribution to valence
            k_avoid: Weight for MBON avoid contribution to valence (subtracted)
            k_dan: Weight for DAN reinforcement contribution to valence
            k_arousal: Weight for arousal signal scaling
            arousal_baseline: Baseline arousal (Hz) for normalization
        """
        self.k_approach = k_approach
        self.k_avoid = k_avoid
        self.k_dan = k_dan
        self.k_arousal = k_arousal
        self.arousal_baseline = arousal_baseline
        
    def to_core_affect(self, rates: MBONDANRates) -> CoreAffect:
        """Convert MBON/DAN rates to CoreAffect (valence, arousal).
        
        Args:
            rates: MBON/DAN firing rates from fly circuit
            
        Returns:
            CoreAffect with valence and arousal in [-1, 1]
        """
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
    
    def to_appraisal_vector(
        self,
        rates: MBONDANRates,
        dominance: float | None = None,
        unpredictability: float | None = None,
    ) -> AppraisalVector:
        """Convert MBON/DAN rates to full AppraisalVector.
        
        Args:
            rates: MBON/DAN firing rates
            dominance: Optional dominance dimension (defaults to neutral 0.5)
            unpredictability: Optional unpredictability (defaults to neutral 0.5)
            
        Returns:
            AppraisalVector with valence, arousal, dominance, unpredictability
        """
        core = self.to_core_affect(rates)
        
        # Dominance: heuristic from approach drive (high approach → high dominance)
        if dominance is None:
            approach_drive = rates.mbon_approach - rates.mbon_avoid
            dominance = float(np.tanh(0.01 * approach_drive))
            dominance = (dominance + 1) / 2  # Map [-1,1] to [0,1]
        
        # Unpredictability: heuristic from arousal and DAN variance
        # (high arousal + variable reinforcement → high unpredictability)
        if unpredictability is None:
            unpredictability = 0.5  # Default neutral for now
        
        return AppraisalVector(
            valence=core.valence,
            arousal=core.arousal,
            dominance=dominance,
            unpredictability=unpredictability,
        )
    
    def get_approach_avoid(self, rates: MBONDANRates) -> float:
        """Get approach/avoid drive for policy decisions.
        
        Args:
            rates: MBON/DAN firing rates
            
        Returns:
            Approach drive in Hz (positive=approach, negative=avoid)
        """
        return rates.mbon_approach - rates.mbon_avoid
    
    def is_approach_dominant(self, rates: MBONDANRates, threshold: float = 0.0) -> bool:
        """Check if approach drive exceeds threshold.
        
        Args:
            rates: MBON/DAN firing rates
            threshold: Approach threshold (Hz)
            
        Returns:
            True if approach drive > threshold
        """
        return self.get_approach_avoid(rates) > threshold
