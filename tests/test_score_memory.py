"""
Unit tests for ScoreMemory (GCL-based Causal Blackboard).
"""

import unittest
from counterpoint.core.score_memory import ScoreMemory


class TestScoreMemory(unittest.TestCase):
    def setUp(self):
        self.score = ScoreMemory(hidden_dim=32, schema_dim=8, seed=123)

    def test_cumulative_pooling_matches_expectation(self):
        # Step 1
        vec1 = [1.0] * 32
        ctx1 = self.score.update_cumulative(vec1)
        self.assertEqual(len(ctx1), 32)
        self.assertAlmostEqual(ctx1[0], 1.0)
        self.assertEqual(self.score.step_count, 1)

        # Step 2
        vec2 = [3.0] * 32
        ctx2 = self.score.update_cumulative(vec2)
        # Average of 1.0 and 3.0 is 2.0
        self.assertAlmostEqual(ctx2[0], 2.0)
        self.assertEqual(self.score.step_count, 2)

    def test_global_schema_projection(self):
        vec = [0.5] * 32
        self.score.update_cumulative(vec)
        schema = self.score.get_global_schema()
        
        self.assertEqual(len(schema), 8)
        # Tanh outputs must be in [-1.0, 1.0]
        for val in schema:
            self.assertTrue(-1.0 <= val <= 1.0)

    def test_adaptive_gate_bounds(self):
        vec = [0.2] * 32
        self.score.update_cumulative(vec)
        
        gate = self.score.compute_adaptive_gate(vec)
        self.assertIsInstance(gate, float)
        self.assertTrue(0.0 <= gate <= 1.0)

    def test_schema_reflection(self):
        vec = [0.4] * 32
        self.score.update_cumulative(vec)
        
        enhanced, gate = self.score.reflect_schema(vec)
        self.assertEqual(len(enhanced), 32)
        self.assertTrue(0.0 <= gate <= 1.0)

    def test_reset(self):
        vec = [1.0] * 32
        self.score.update_cumulative(vec)
        self.assertEqual(self.score.step_count, 1)
        
        self.score.reset()
        self.assertEqual(self.score.step_count, 0)
        self.assertIsNone(self.score.cached_cumsum)
        self.assertEqual(self.score.cached_counts, 0.0)


if __name__ == "__main__":
    unittest.main()
