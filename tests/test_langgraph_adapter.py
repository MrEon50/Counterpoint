"""
Unit tests for LangGraphCounterpointNode adapter.
"""

import unittest
from counterpoint.mesh import CounterpointMesh
from counterpoint.adapters.langgraph_adapter import LangGraphCounterpointNode


class TestLangGraphAdapter(unittest.TestCase):
    def setUp(self):
        self.mesh = CounterpointMesh(hidden_dim=16, schema_dim=4, opposition_threshold=0.25)

    def test_node_execution_and_routing(self):
        def pass_verifier(content: str) -> bool:
            return True

        node = LangGraphCounterpointNode(self.mesh, verifier_fn=pass_verifier)

        # Initial graph state
        initial_state = {
            "coder_output": "def foo(): return 1",
            "critic_output": "def foo(): return 2 # fix"
        }

        # Run node
        next_state = node(initial_state)

        self.assertIn("counterpoint_bar", next_state)
        self.assertIn("counterpoint_is_resolved", next_state)
        self.assertEqual(next_state["counterpoint_bar"], 1)

        # Routing decision
        decision = node.should_continue(next_state)
        self.assertIn(decision, ["resolved", "continue"])


if __name__ == "__main__":
    unittest.main()
