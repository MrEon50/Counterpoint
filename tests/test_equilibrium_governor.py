"""
Unit tests for EquilibriumGovernor and Goodhart Guard.
"""

import unittest
from counterpoint.core.equilibrium_governor import EquilibriumGovernor


class TestEquilibriumGovernor(unittest.TestCase):
    def setUp(self):
        self.gov = EquilibriumGovernor(target_harmony=0.0, tolerance=0.30)

    def test_initial_harmony(self):
        # Default rules include creative (+0.8), test safety (-0.9), budget (-0.6)
        phi = self.gov.calculate_system_harmony()
        self.assertTrue(-1.0 <= phi <= 1.0)

    def test_balance_rules_when_over_expansive(self):
        # Artificially boost expansion rule weight to +10.0
        self.gov.rules["creative_exploration"].weight = 10.0
        self.gov.rules["unit_test_safety"].weight = 0.5
        
        phi_before = self.gov.calculate_system_harmony()
        self.assertTrue(phi_before > 0.30)  # Beyond positive tolerance

        self.gov.balance_weights()
        
        # Protection weight should have increased, expansion should have decreased
        self.assertTrue(self.gov.rules["unit_test_safety"].weight > 0.5)
        self.assertTrue(self.gov.rules["creative_exploration"].weight < 10.0)

    def test_stagnation_detection(self):
        # Feed identical phi values to trigger low variance
        for _ in range(20):
            self.gov._phi_history.append(0.0)

        stagnant, msg = self.gov.detect_stagnation_and_perturb()
        self.assertTrue(stagnant)
        self.assertIsNotNone(msg)

    def test_goodhart_gap_quarantine(self):
        # Simulate reward hacking: 0 tension (pretending harmony) while failing 100% of tasks
        for _ in range(10):
            self.gov.record_step_outcome(collective_tension=0.05, success=False)

        gamma, is_hacking = self.gov.check_goodhart_gap()
        self.assertTrue(is_hacking)
        self.assertTrue(gamma > 0.35)

        # Creative exploration rule should be quarantined
        creative_rule = self.gov.rules["creative_exploration"]
        self.assertTrue(creative_rule.quarantined)
        self.assertIn("Goodhart", creative_rule.quarantine_reason)


if __name__ == "__main__":
    unittest.main()
