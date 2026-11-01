"""
Unit tests for ResolutionEngine (Dialectical Synthesis Gate & R(A, B) -> O').
"""

import unittest
from counterpoint.core.types import AgentState
from counterpoint.core.resolution_engine import ResolutionEngine


class TestResolutionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ResolutionEngine(opposition_threshold=0.30, complexity_penalty=0.05)

    def test_beauty_index_calculation(self):
        # f_AB = 0.4, depth=2, n_act=2
        b = self.engine.compute_beauty_index(f_AB=0.4, depth_a=2.0, depth_b=2.0, n_active=2)
        # B = (0.5 * (2 + 2)) / (0.4 + 0.05*2 + 0.1) = 2.0 / 0.6 = 3.3333
        self.assertAlmostEqual(b, 3.3333, places=3)

    def test_threshold_blocks_low_opposition(self):
        state_a = AgentState(voice_id="A", content="Code version 1")
        state_b = AgentState(voice_id="B", content="Code version 1 slightly modified")
        
        # Low opposition (0.15 < 0.30)
        res = self.engine.attempt_resolution(
            voice_a="A",
            state_a=state_a,
            voice_b="B",
            state_b=state_b,
            opposition_score=0.15,
            collective_tension_before=0.5
        )
        self.assertFalse(res.triggered)
        self.assertFalse(res.accepted)

    def test_synthesis_accepted_when_anchor_passes(self):
        state_a = AgentState(
            voice_id="A",
            content="def compute(x): return x + 1",
            latent_vector=[1.0, 0.0, 0.0]
        )
        state_b = AgentState(
            voice_id="B",
            content="Error: function should return x * 2",
            latent_vector=[-1.0, 0.0, 0.0]
        )

        def mock_pytest_pass(content: str) -> bool:
            return True

        res = self.engine.attempt_resolution(
            voice_a="A",
            state_a=state_a,
            voice_b="B",
            state_b=state_b,
            opposition_score=0.85,
            collective_tension_before=0.6,
            verifier_fn=mock_pytest_pass
        )

        self.assertTrue(res.triggered)
        self.assertEqual(res.external_verdict, 1)
        self.assertTrue(res.delta_syn > 0)
        self.assertTrue(res.accepted)
        self.assertIsNotNone(res.synthesized_content)
        self.assertIsNotNone(res.synthesized_vector)

    def test_synthesis_rejected_when_anchor_fails(self):
        state_a = AgentState(voice_id="A", content="broken solution A")
        state_b = AgentState(voice_id="B", content="broken solution B")

        def mock_pytest_fail(content: str) -> bool:
            return False

        res = self.engine.attempt_resolution(
            voice_a="A",
            state_a=state_a,
            voice_b="B",
            state_b=state_b,
            opposition_score=0.75,
            collective_tension_before=0.6,
            verifier_fn=mock_pytest_fail
        )

        self.assertTrue(res.triggered)
        self.assertEqual(res.external_verdict, 0)
        self.assertFalse(res.accepted)


if __name__ == "__main__":
    unittest.main()
