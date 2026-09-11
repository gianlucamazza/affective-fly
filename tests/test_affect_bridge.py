"""Tests for affect bridge."""

import numpy as np
import pytest

from affective_fly.affect_bridge import AffectBridge
from affective_fly.fly_circuit import MBONDANRates


class TestAffectBridge:
    """Test MBON/DAN to CoreAffect mapping."""
    
    def test_initialization(self):
        bridge = AffectBridge()
        assert bridge.k_approach > 0
        assert bridge.k_avoid > 0
    
    def test_positive_valence(self):
        bridge = AffectBridge()
        rates = MBONDANRates(
            mbon_approach=80.0,  # High approach
            mbon_avoid=20.0,     # Low avoid
            dan_reinforcement=10.0,  # Positive reinforcement
            arousal_signal=40.0,
        )
        affect = bridge.to_core_affect(rates)
        
        # High approach + positive DAN → positive valence
        assert affect.valence > 0
        assert -1.0 <= affect.valence <= 1.0
        assert -1.0 <= affect.arousal <= 1.0
    
    def test_negative_valence(self):
        bridge = AffectBridge()
        rates = MBONDANRates(
            mbon_approach=20.0,  # Low approach
            mbon_avoid=80.0,     # High avoid
            dan_reinforcement=-10.0,  # Negative reinforcement
            arousal_signal=40.0,
        )
        affect = bridge.to_core_affect(rates)
        
        # High avoid + negative DAN → negative valence
        assert affect.valence < 0
    
    def test_arousal_scaling(self):
        bridge = AffectBridge()
        
        # Low arousal
        rates_low = MBONDANRates(
            mbon_approach=50.0,
            mbon_avoid=50.0,
            dan_reinforcement=0.0,
            arousal_signal=10.0,  # Low
        )
        affect_low = bridge.to_core_affect(rates_low)
        
        # High arousal
        rates_high = MBONDANRates(
            mbon_approach=50.0,
            mbon_avoid=50.0,
            dan_reinforcement=0.0,
            arousal_signal=80.0,  # High
        )
        affect_high = bridge.to_core_affect(rates_high)
        
        # High arousal signal → higher arousal
        assert affect_high.arousal > affect_low.arousal
    
    def test_to_appraisal_vector(self):
        bridge = AffectBridge()
        rates = MBONDANRates(
            mbon_approach=70.0,
            mbon_avoid=30.0,
            dan_reinforcement=5.0,
            arousal_signal=50.0,
        )
        appraisal = bridge.to_appraisal_vector(rates)
        
        assert hasattr(appraisal, 'valence')
        assert hasattr(appraisal, 'arousal')
        assert hasattr(appraisal, 'dominance')
        assert hasattr(appraisal, 'unpredictability')
        assert 0.0 <= appraisal.dominance <= 1.0
    
    def test_get_approach_avoid(self):
        bridge = AffectBridge()
        rates = MBONDANRates(
            mbon_approach=70.0,
            mbon_avoid=30.0,
            dan_reinforcement=0.0,
            arousal_signal=40.0,
        )
        
        approach_drive = bridge.get_approach_avoid(rates)
        assert approach_drive == 40.0  # 70 - 30
    
    def test_is_approach_dominant(self):
        bridge = AffectBridge()
        
        # Approach dominant
        rates_approach = MBONDANRates(
            mbon_approach=70.0,
            mbon_avoid=20.0,
            dan_reinforcement=0.0,
            arousal_signal=40.0,
        )
        assert bridge.is_approach_dominant(rates_approach, threshold=0.0)
        
        # Avoid dominant
        rates_avoid = MBONDANRates(
            mbon_approach=20.0,
            mbon_avoid=70.0,
            dan_reinforcement=0.0,
            arousal_signal=40.0,
        )
        assert not bridge.is_approach_dominant(rates_avoid, threshold=0.0)
