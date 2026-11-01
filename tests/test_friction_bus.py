"""
Unit tests for FrictionBus and tension dynamics.
"""

import unittest
from counterpoint.core.types import FrictionVector, AgentState
from counterpoint.core.friction_bus import FrictionBus


class TestFrictionBus(unittest.TestCase):
    def setUp(self):
        self.bus = FrictionBus(kappa_N=0.25, delta=0.40, kappa=0.10)

    def test_friction_vector_norm(self):
        # Case 1: All zeros
        f0 = FrictionVector(0, 0, 0, 0)
        self.assertEqual(f0.compute_norm(kappa_N=0.25), 0.0)

        # Case 2: All ones
        f1 = FrictionVector(1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(f1.compute_norm(kappa_N=0.25), 1.0, places=4)

        # Case 3: Acute point crisis (0.8, 0, 0, 0) vs dispersed (0.2, 0.2, 0.2, 0.2)
        f_acute = FrictionVector(0.8, 0.0, 0.0, 0.0)
        norm_acute = f_acute.compute_norm(kappa_N=0.0) # RMS
        self.assertAlmostEqual(norm_acute, 0.4, places=4)

    def test_stability_condition_enforcement(self):
        # If delta <= 3 * kappa, bus automatically clamps kappa to maintain stability
        bus_unstable = FrictionBus(delta=0.30, kappa=0.15)
        # Should be auto-tuned to < 0.30 / 3 = 0.10
        self.assertTrue(bus_unstable.delta > 3.0 * bus_unstable.kappa)

    def test_pairwise_opposition(self):
        # Opposite unit vectors: [1, 0, 0] and [-1, 0, 0] -> cos = -1 -> f_AB = 1.0
        state_a = AgentState(voice_id="A", latent_vector=[1.0, 0.0, 0.0])
        state_b = AgentState(voice_id="B", latent_vector=[-1.0, 0.0, 0.0])
        
        self.bus.register_step("A", state_a)
        self.bus.register_step("B", state_b)
        
        opp = self.bus.compute_pairwise_opposition("A", "B")
        self.assertAlmostEqual(opp, 1.0, places=4)

        # Identical vectors: [1, 0, 0] and [1, 0, 0] -> cos = 1 -> f_AB = 0.0
        state_c = AgentState(voice_id="C", latent_vector=[1.0, 0.0, 0.0])
        self.bus.register_step("C", state_c)
        opp_same = self.bus.compute_pairwise_opposition("A", "C")
        self.assertAlmostEqual(opp_same, 0.0, places=4)

    def test_cross_surprise_tv(self):
        # Total Variation between identical distributions is 0
        state_a = AgentState(voice_id="A", policy_logits=[1.0, 0.0])
        state_b = AgentState(voice_id="B", policy_logits=[1.0, 0.0])
        self.bus.register_step("A", state_a)
        self.bus.register_step("B", state_b)
        
        tv_zero = self.bus.compute_cross_surprise("A", "B")
        self.assertAlmostEqual(tv_zero, 0.0, places=4)

        # Total Variation between divergent distributions is > 0
        state_div = AgentState(voice_id="C", policy_logits=[-5.0, 5.0])
        self.bus.register_step("C", state_div)
        tv_high = self.bus.compute_cross_surprise("A", "C")
        self.assertTrue(tv_high > 0.5)

    def test_collective_tension_dispersion(self):
        # Two agents with different tensions should have collective tension > mean norm
        s1 = AgentState(voice_id="A", friction=FrictionVector(0.8, 0.0, 0.0, 0.0))
        s2 = AgentState(voice_id="B", friction=FrictionVector(0.1, 0.0, 0.0, 0.0))
        self.bus.register_step("A", s1)
        self.bus.register_step("B", s2)

        n_col = self.bus.compute_collective_tension()
        norm_a = s1.friction.compute_norm(self.bus.kappa_N)
        norm_b = s2.friction.compute_norm(self.bus.kappa_N)
        mean_norm = (norm_a + norm_b) / 2.0
        
        # n_col includes gamma_couple * |norm_a - norm_b|
        self.assertTrue(n_col > mean_norm)


if __name__ == "__main__":
    unittest.main()
