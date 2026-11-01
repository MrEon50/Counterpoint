"""
Unit tests for EpistemicLedger (Dynamic Authority & Anti-Compromise Truth Tracker).
"""

import unittest
from counterpoint.core.epistemic_ledger import EpistemicLedger


class TestEpistemicLedgerCalibration(unittest.TestCase):
    """Tests for ground-truth calibration probing."""

    def setUp(self):
        self.ledger = EpistemicLedger(
            initial_authority=0.50,
            accuracy_bonus=0.10,
            accuracy_penalty=0.12,
            overconfidence_beta=0.15,
            quarantine_threshold=0.15
        )
        self.ledger.register_voice("Coder")
        self.ledger.register_voice("Reviewer")

    def test_initial_authority(self):
        self.assertAlmostEqual(self.ledger.get_authority("Coder"), 0.50)
        self.assertAlmostEqual(self.ledger.get_authority("Reviewer"), 0.50)

    def test_correct_answer_increases_authority(self):
        self.ledger.run_calibration_probe(
            voice_id="Coder",
            question_id="q1",
            expected_answer="42",
            voice_answer="42",
            confidence=0.9
        )
        # Should be 0.50 + 0.10 = 0.60
        self.assertAlmostEqual(self.ledger.get_authority("Coder"), 0.60)

    def test_wrong_answer_decreases_authority(self):
        self.ledger.run_calibration_probe(
            voice_id="Reviewer",
            question_id="q1",
            expected_answer="42",
            voice_answer="99",
            confidence=0.5
        )
        # Penalty = 0.12 + 0.15 * 0.5 = 0.195
        # Authority = 0.50 - 0.195 = 0.305
        self.assertAlmostEqual(self.ledger.get_authority("Reviewer"), 0.305)

    def test_overconfidence_penalty_scales_with_confidence(self):
        """A voice that says '99' with confidence 1.0 gets a harsher penalty
        than one that says '99' with confidence 0.1."""
        self.ledger.run_calibration_probe(
            voice_id="Coder",
            question_id="q1",
            expected_answer="42",
            voice_answer="99",
            confidence=1.0
        )
        w_high_conf = self.ledger.get_authority("Coder")

        self.ledger.register_voice("Humble")
        self.ledger.run_calibration_probe(
            voice_id="Humble",
            question_id="q1",
            expected_answer="42",
            voice_answer="99",
            confidence=0.1
        )
        w_low_conf = self.ledger.get_authority("Humble")

        # High confidence wrong answer should result in lower authority
        self.assertLess(w_high_conf, w_low_conf)

    def test_quarantine_on_low_authority(self):
        """Multiple wrong answers should trigger epistemic quarantine."""
        self.assertFalse(self.ledger.is_quarantined("Reviewer"))

        # Simulate 3 consecutive wrong answers with high confidence
        for i in range(3):
            self.ledger.run_calibration_probe(
                voice_id="Reviewer",
                question_id=f"q{i}",
                expected_answer="correct",
                voice_answer="wrong",
                confidence=0.9
            )

        # Authority should be very low → quarantined
        self.assertTrue(self.ledger.is_quarantined("Reviewer"))
        self.assertLessEqual(self.ledger.get_authority("Reviewer"), 0.15)

    def test_accuracy_rate_tracking(self):
        # 2 correct, 1 wrong
        self.ledger.run_calibration_probe("Coder", "q1", "a", "a", 0.5)
        self.ledger.run_calibration_probe("Coder", "q2", "b", "b", 0.5)
        self.ledger.run_calibration_probe("Coder", "q3", "c", "x", 0.5)

        rate = self.ledger.get_accuracy_rate("Coder")
        self.assertAlmostEqual(rate, 2.0 / 3.0, places=3)

    def test_calibration_suite(self):
        probes = [
            {"question_id": "q1", "question": "What is 2+2?", "expected_answer": "4"},
            {"question_id": "q2", "question": "What is 3*3?", "expected_answer": "9"},
        ]

        def answer_fn(question):
            if "2+2" in question:
                return "4", 0.95
            return "9", 0.8

        results = self.ledger.run_calibration_suite("Coder", probes, answer_fn)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.correct for r in results))


class TestAsymmetricTruthGravity(unittest.TestCase):
    """Tests for asymmetric weight computation and anti-compromise logic."""

    def setUp(self):
        self.ledger = EpistemicLedger(
            initial_authority=0.50,
            dominance_threshold=0.40
        )
        self.ledger.register_voice("Expert")
        self.ledger.register_voice("Novice")

    def test_equal_authority_gives_balanced_weights(self):
        w_a, w_b, rejected = self.ledger.compute_asymmetric_weights("Expert", "Novice")
        self.assertAlmostEqual(w_a, 0.5)
        self.assertAlmostEqual(w_b, 0.5)
        self.assertFalse(rejected)

    def test_large_gap_triggers_compromise_rejection(self):
        # Boost Expert authority to 0.90 via calibration
        self.ledger._authority["Expert"] = 0.90
        # Drop Novice authority to 0.20
        self.ledger._authority["Novice"] = 0.20

        w_a, w_b, rejected = self.ledger.compute_asymmetric_weights("Expert", "Novice")
        # Gap = 0.70 > 0.40 threshold → compromise rejected
        self.assertTrue(rejected)
        self.assertAlmostEqual(w_a, 1.0)
        self.assertAlmostEqual(w_b, 0.0)

    def test_moderate_gap_gives_soft_asymmetry(self):
        self.ledger._authority["Expert"] = 0.65
        self.ledger._authority["Novice"] = 0.35

        w_a, w_b, rejected = self.ledger.compute_asymmetric_weights("Expert", "Novice")
        # Gap = 0.30 < 0.40 → soft asymmetry, not rejected
        self.assertFalse(rejected)
        self.assertGreater(w_a, w_b)
        self.assertAlmostEqual(w_a + w_b, 1.0, places=3)

    def test_quarantined_voice_gets_zero_weight(self):
        self.ledger._quarantined["Novice"] = True
        w_a, w_b, rejected = self.ledger.compute_asymmetric_weights("Expert", "Novice")
        self.assertTrue(rejected)
        self.assertAlmostEqual(w_a, 1.0)
        self.assertAlmostEqual(w_b, 0.0)


class TestEpistemicLeader(unittest.TestCase):
    """Tests for leader selection (Litmus Probe privilege)."""

    def setUp(self):
        self.ledger = EpistemicLedger()
        self.ledger.register_voice("Alpha")
        self.ledger.register_voice("Beta")
        self.ledger.register_voice("Gamma")

    def test_leader_is_highest_authority(self):
        self.ledger._authority["Alpha"] = 0.80
        self.ledger._authority["Beta"] = 0.50
        self.ledger._authority["Gamma"] = 0.30

        leader = self.ledger.get_leader(["Alpha", "Beta", "Gamma"])
        self.assertEqual(leader, "Alpha")

    def test_quarantined_voice_cannot_be_leader(self):
        self.ledger._authority["Alpha"] = 0.80
        self.ledger._authority["Beta"] = 0.50
        self.ledger._quarantined["Alpha"] = True

        leader = self.ledger.get_leader(["Alpha", "Beta", "Gamma"])
        self.assertEqual(leader, "Beta")

    def test_empty_list_returns_none(self):
        self.assertIsNone(self.ledger.get_leader([]))


class TestLedgerDiagnostics(unittest.TestCase):
    """Tests for the status/diagnostics API."""

    def test_get_status(self):
        ledger = EpistemicLedger()
        ledger.register_voice("V1")
        ledger.register_voice("V2")
        ledger.run_calibration_probe("V1", "q1", "yes", "yes", 0.8)

        status = ledger.get_status()
        self.assertIn("V1", status)
        self.assertIn("V2", status)
        self.assertGreater(status["V1"]["authority"], 0.5)
        self.assertEqual(status["V1"]["calibration_probes"], 1)


if __name__ == "__main__":
    unittest.main()
