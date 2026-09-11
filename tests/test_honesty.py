"""Tests for honesty layer."""

import pytest

from affective_fly.honesty import HonestyLayer, human_label_readout
from affective_fly.mood_field import CoreAffect


class TestHonestyLayer:
    """Test human emotion label mapping."""
    
    def test_initialization(self):
        layer = HonestyLayer()
        assert layer.arousal_threshold > 0
    
    def test_label_excited(self):
        layer = HonestyLayer()
        affect = CoreAffect(valence=0.8, arousal=0.7)
        
        readout = layer.label(affect)
        
        # High valence + high arousal → excited
        assert readout.primary_label == "excited"
        assert readout.confidence > 0.5
        assert "interpretive" in readout.disclaimer.lower() or "approximation" in readout.disclaimer.lower()
    
    def test_label_calm(self):
        layer = HonestyLayer()
        affect = CoreAffect(valence=0.6, arousal=-0.2)
        
        readout = layer.label(affect)
        
        # Positive valence + low arousal → calm or content
        assert readout.primary_label in ["calm", "content"]
    
    def test_label_anxious(self):
        layer = HonestyLayer()
        affect = CoreAffect(valence=-0.7, arousal=0.8)
        
        readout = layer.label(affect)
        
        # Negative valence + high arousal → anxious
        assert readout.primary_label == "anxious"
    
    def test_label_sad(self):
        layer = HonestyLayer()
        affect = CoreAffect(valence=-0.6, arousal=-0.5)
        
        readout = layer.label(affect)
        
        # Negative valence + low arousal → sad or lethargic
        assert readout.primary_label in ["sad", "lethargic"]
    
    def test_label_neutral(self):
        layer = HonestyLayer()
        affect = CoreAffect(valence=0.0, arousal=0.0)
        
        readout = layer.label(affect)
        
        # Near origin → low confidence
        assert readout.confidence < 0.3
    
    def test_confidence_scales_with_distance(self):
        layer = HonestyLayer()
        
        # Near neutral
        affect_near = CoreAffect(valence=0.1, arousal=0.1)
        readout_near = layer.label(affect_near)
        
        # Far from neutral
        affect_far = CoreAffect(valence=0.9, arousal=0.9)
        readout_far = layer.label(affect_far)
        
        # Further from neutral → higher confidence
        assert readout_far.confidence > readout_near.confidence
    
    def test_label_with_circumplex(self):
        layer = HonestyLayer()
        affect = CoreAffect(valence=0.7, arousal=0.5)
        
        label_str = layer.label_with_circumplex(affect)
        
        # Should include label, coordinates, and interpretive marker
        assert "v=" in label_str
        assert "a=" in label_str
        assert "interpretive" in label_str.lower()
    
    def test_human_label_readout_convenience(self):
        affect = CoreAffect(valence=0.8, arousal=0.6)
        
        label_with = human_label_readout(affect, include_disclaimer=True)
        label_without = human_label_readout(affect, include_disclaimer=False)
        
        # With disclaimer should be longer
        assert len(label_with) > len(label_without)
        assert "fly circuit" in label_with.lower() or "approximation" in label_with.lower()
