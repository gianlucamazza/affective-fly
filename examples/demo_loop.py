#!/usr/bin/env python3
"""Demo: Basic affective loop with mock fly circuit.

Demonstrates:
1. Sensory frames → fly circuit → CoreAffect
2. Mood field slow integration
3. Emotional memory encoding/retrieval
4. Policy decisions based on affect + memories
5. Journal logging with circumplex data
"""

import numpy as np

from affective_fly import (
    AffectiveLoop,
    SensoryFrame,
)


def main():
    """Run basic affective loop demo."""
    print("=" * 70)
    print("Affective Fly: Basic Loop Demo")
    print("=" * 70)
    
    # Create affective loop with mock circuit
    loop = AffectiveLoop(agent_id="demo-fly", mood_half_life=100)
    
    # Simulate 50 timesteps with varying sensory input
    print("\nRunning 50-step simulation...\n")
    
    tickers = ["DOGE", "PEPE", "SHIB", "BTC", "ETH"]
    
    for step in range(50):
        # Generate sensory features (positive trend, then negative crash)
        if step < 25:
            # Positive market: features drift positive
            features = np.random.randn(10) + 0.5 * (step / 25)
            pnl = step * 10
            market = "rising"
        else:
            # Negative crash: features plunge negative
            features = np.random.randn(10) - 1.0 * ((step - 25) / 25)
            pnl = 250 - (step - 25) * 15
            market = "crashing"
        
        ticker = tickers[step % len(tickers)]
        
        frame = SensoryFrame(
            features=features,
            context=f"ticker: {ticker}, market: {market}",
            metadata={"ticker": ticker, "pnl": pnl, "step": step}
        )
        
        # Process frame
        decision = loop.step(frame)
        
        # Print selected steps
        if step % 10 == 0 or step == 24 or step == 25:
            mood = loop.get_mood()
            print(f"Step {step:2d} | {market:8s} | "
                  f"action: {decision.action.value:12s} | "
                  f"mood_v: {mood.valence:+.2f} | "
                  f"conf: {decision.confidence:.2f}")
            print(f"         Rationale: {decision.rationale}")
    
    # Print journal summary
    loop.journal.print_summary()
    
    # Export journal to JSON
    print("Exporting journal to demo_journal.json...")
    loop.journal.export_json("demo_journal.json")
    print("Done! Check demo_journal.json for full circumplex data.\n")


if __name__ == "__main__":
    main()
