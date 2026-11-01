"""
Unit tests for LitmusEngine (Leader's Falsification Privilege & Socratic Probe).
"""

import unittest
from counterpoint.core.litmus_engine import LitmusEngine, LitmusResult


class TestLitmusEngineBasics(unittest.TestCase):
    """Tests for basic Litmus Probe registration and execution."""

    def setUp(self):
        self.engine = LitmusEngine()

    def test_no_probe_auto_passes(self):
        result = self.engine.execute_probe(
            leader_voice_id="Leader",
            target_voice_id="Rival",
            content_to_test="def foo(): return 42"
        )
        self.assertTrue(result.probe_passed)
        self.assertIn("auto-pass", result.message)

    def test_probe_passes_when_content_is_correct(self):
        # Leader registers a probe that checks if 'return' is in the code
        self.engine.register_probe(
            leader_voice_id="Leader",
            probe_fn=lambda content: "return" in content,
            description="Must contain a return statement",
            probe_name="has_return"
        )

        result = self.engine.execute_probe(
            leader_voice_id="Leader",
            target_voice_id="Rival",
            content_to_test="def foo(): return 42"
        )
        self.assertTrue(result.probe_passed)
        self.assertIn("PASSED", result.message)

    def test_probe_fails_when_content_is_wrong(self):
        # Leader's edge case: function must handle negative input
        self.engine.register_probe(
            leader_voice_id="Leader",
            probe_fn=lambda content: "if x < 0" in content or "abs(" in content,
            description="Must handle negative input",
            probe_name="handles_negative"
        )

        result = self.engine.execute_probe(
            leader_voice_id="Leader",
            target_voice_id="Rival",
            content_to_test="def foo(x): return x * 2"
        )
        self.assertFalse(result.probe_passed)
        self.assertIn("FAILED", result.message)

    def test_probe_exception_treated_as_failure(self):
        # Probe that throws an exception
        def buggy_probe(content):
            raise ValueError("Probe crashed!")

        self.engine.register_probe(
            leader_voice_id="Leader",
            probe_fn=buggy_probe,
            description="Buggy probe"
        )

        result = self.engine.execute_probe(
            leader_voice_id="Leader",
            target_voice_id="Rival",
            content_to_test="some content"
        )
        self.assertFalse(result.probe_passed)
        self.assertIn("exception", result.message)


class TestLitmusProbeHistory(unittest.TestCase):
    """Tests for Litmus Probe audit trail and statistics."""

    def setUp(self):
        self.engine = LitmusEngine()

    def test_history_tracking(self):
        self.engine.register_probe("L1", lambda c: True, "passes")
        self.engine.execute_probe("L1", "R1", "content A")
        self.engine.execute_probe("L1", "R2", "content B")

        history = self.engine.get_history()
        self.assertEqual(len(history), 2)

    def test_stats_calculation(self):
        # Register probe that only passes content containing "correct"
        self.engine.register_probe(
            "Leader",
            lambda c: "correct" in c,
            "Must contain 'correct'"
        )

        self.engine.execute_probe("Leader", "R1", "correct solution")
        self.engine.execute_probe("Leader", "R2", "wrong solution")
        self.engine.execute_probe("Leader", "R3", "also correct answer")

        stats = self.engine.get_stats()
        self.assertEqual(stats["total_probes"], 3)
        self.assertAlmostEqual(stats["pass_rate"], 2 / 3, places=3)
        self.assertEqual(stats["eliminations"], 1)

    def test_clear_probes(self):
        self.engine.register_probe("L1", lambda c: True, "test")
        self.assertTrue(self.engine.has_active_probe("L1"))

        self.engine.clear_probes()
        self.assertFalse(self.engine.has_active_probe("L1"))


class TestLitmusRealWorldScenario(unittest.TestCase):
    """Realistic scenario: Leader catches a bug that naive averaging would miss."""

    def test_earth_shape_scenario(self):
        """
        Agent A: "Earth is roughly spherical" (correct)
        Agent B: "Earth is flat" (incorrect)
        
        Without Litmus: naive average -> "Earth is a hemisphere" (absurd)
        With Litmus: Leader creates a test -> "Does the proposal explain
        why ships disappear bottom-first over the horizon?"
        """
        engine = LitmusEngine()

        # Leader (Agent A, more credible) creates a falsification test
        def horizon_test(content: str) -> bool:
            """If the proposal can explain ships disappearing bottom-first,
            it's consistent with a curved surface."""
            lower = content.lower()
            return "sphere" in lower or "curved" in lower or "round" in lower

        engine.register_probe(
            leader_voice_id="AgentA",
            probe_fn=horizon_test,
            description="Must explain horizon effect (ships disappearing bottom-first)",
            probe_name="horizon_test"
        )

        # Test Agent B's flat earth proposal
        result_b = engine.execute_probe(
            leader_voice_id="AgentA",
            target_voice_id="AgentB",
            content_to_test="The Earth is flat like a disc, ships just get smaller"
        )
        self.assertFalse(result_b.probe_passed)

        # Test the naive compromise "hemisphere" — also fails!
        result_compromise = engine.execute_probe(
            leader_voice_id="AgentA",
            target_voice_id="Compromise",
            content_to_test="The Earth is a hemisphere, half sphere half flat"
        )
        # This actually passes because it contains "sphere" — but let's make
        # a stricter test
        engine.clear_probes()

        def strict_horizon_test(content: str) -> bool:
            lower = content.lower()
            # Must mention curvature AND not mention "flat"
            return ("sphere" in lower or "curved" in lower) and "flat" not in lower

        engine.register_probe("AgentA", strict_horizon_test, "Strict curvature test")

        # Now the compromise fails too
        result_strict = engine.execute_probe(
            "AgentA", "Compromise",
            "The Earth is a hemisphere, half sphere half flat"
        )
        self.assertFalse(result_strict.probe_passed)

        # But the correct answer passes
        result_correct = engine.execute_probe(
            "AgentA", "Truth",
            "The Earth is roughly a sphere with an oblate shape due to rotation"
        )
        self.assertTrue(result_correct.probe_passed)


if __name__ == "__main__":
    unittest.main()
