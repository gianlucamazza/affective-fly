"""Tests for policy."""


from affective_fly.mood_field import MoodState
from affective_fly.policy import Action, Policy, PolicyDecision


def test_policy_decision():
    """Test PolicyDecision dataclass."""
    decision = PolicyDecision(
        action=Action.CLICK,
        target="button",
        confidence=0.8,
        mood_valence=0.5,
        mood_arousal=0.3,
        approach_tendency=0.6,
        reason="test",
    )
    assert decision.action == Action.CLICK
    assert decision.target == "button"
    assert decision.confidence == 0.8


def test_policy_avoidance():
    """Test policy triggers avoidance for low approach."""
    policy = Policy(threshold_avoid=-0.3)
    mood = MoodState(valence=0.0, arousal=0.0, approach_tendency=-0.5)

    decision = policy.decide(mood, [], {})

    assert decision.action == Action.SKIP
    assert "avoidance" in decision.reason.lower() or "approach" in decision.reason.lower()


def test_policy_low_arousal():
    """WAIT when arousal is at or below the default calm threshold (0.0).

    Arousal is on [0, 1] (CoreAffect). The pre-v0.2.5 values
    ``threshold_calm=-0.5`` and ``arousal=-0.7`` were unreachable after
    ``set_affect()`` and are not used here.
    """
    policy = Policy()  # default threshold_calm=0.0
    # Favorable valence/approach so only the arousal gate fires.
    mood = MoodState(valence=0.5, arousal=0.0, approach_tendency=0.3)

    decision = policy.decide(mood, [], {})

    assert decision.action == Action.WAIT
    assert "arousal" in decision.reason.lower()
    assert policy.threshold_calm == 0.0


def test_policy_positive_action():
    """Test policy acts when conditions favorable."""
    policy = Policy(threshold_act=0.2)
    mood = MoodState(valence=0.6, arousal=0.4, approach_tendency=0.5)
    context = {"note_id": "exp-001"}

    decision = policy.decide(mood, [], context)

    assert decision.action in [Action.CLICK, Action.TYPE]
    if decision.action == Action.TYPE:
        assert decision.target == "exp-001"


def test_policy_memory_avoidance():
    """Test policy respects negative memories."""
    policy = Policy()
    mood = MoodState(valence=0.0, arousal=0.0, approach_tendency=0.0)

    # Memory with strong negative valence
    memories = [{"content": "crash", "valence": -0.8, "arousal": 0.5}]

    decision = policy.decide(mood, memories, {})

    assert decision.action == Action.SKIP


def test_policy_default_wait():
    """Test policy defaults to wait with neutral state."""
    policy = Policy()
    mood = MoodState(valence=0.0, arousal=0.0, approach_tendency=0.0)

    decision = policy.decide(mood, [], {})

    assert decision.action == Action.WAIT
