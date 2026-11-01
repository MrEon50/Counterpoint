"""
End-to-end Integration tests for CounterpointMesh.
"""

import unittest
from counterpoint.mesh import CounterpointMesh
from counterpoint.core.types import VoiceRole, FrictionVector


class TestCounterpointMeshIntegration(unittest.TestCase):
    def setUp(self):
        self.mesh = CounterpointMesh(
            hidden_dim=32,
            schema_dim=8,
            opposition_threshold=0.30
        )
        self.coder = self.mesh.register_voice("Coder", role=VoiceRole.GENERATOR)
        self.critic = self.mesh.register_voice("Critic", role=VoiceRole.CRITIC)

    def test_voice_registration(self):
        self.assertIn("Coder", self.mesh.voices)
        self.assertIn("Critic", self.mesh.voices)
        self.assertEqual(self.coder.role, VoiceRole.GENERATOR)
        self.assertEqual(self.critic.role, VoiceRole.CRITIC)

    def test_end_to_end_resolution_workflow(self):
        # Turn 1: Coder writes initial code with a classic off-by-one bug
        state_coder = self.coder.submit(
            content="def bsearch(arr, val):\n    low, high = 0, len(arr)\n    while low < high: ...",
            latent_vector=[1.0] * 32,
            friction=FrictionVector(epistemic=0.2, axiological=0.0, affective=0.3, surprise=0.1)
        )

        # Critic flags the off-by-one error and index out of bounds
        state_critic = self.critic.submit(
            content="Critical bug: high should be len(arr) - 1, and condition should be low <= high to avoid IndexOutOfBounds.",
            latent_vector=[-1.0] * 32, # Opposing semantic vector
            friction=FrictionVector(epistemic=0.85, axiological=0.5, affective=0.6, surprise=0.2)
        )

        # External Programmatic Anchor: simulates pytest
        def mock_verifier(synthesized_content: str) -> bool:
            # Passes only if the synthesized code contains the correction
            return "high" in synthesized_content and "Synthesized" in synthesized_content

        # Execute Counterpoint Bar 1
        bar_record = self.mesh.tick(
            voice_states={"Coder": state_coder, "Critic": state_critic},
            verifier_fn=mock_verifier
        )

        # Verification of Bar 1 outcomes
        self.assertEqual(bar_record.bar_id, 1)
        self.assertTrue(len(bar_record.voices_active) == 2)
        self.assertTrue(bar_record.collective_friction > 0.0)
        self.assertIsNotNone(bar_record.resolution)
        self.assertTrue(bar_record.resolution.triggered)
        self.assertTrue(bar_record.resolution.accepted)
        self.assertTrue(self.mesh.is_resolved())

        # Check status and token economy metrics
        status = self.mesh.get_status()
        self.assertTrue(status["is_resolved"])
        self.assertTrue(status["estimated_tokens_saved"] > 0)

        # Coder checks context for turn 2: sees the accepted resolution
        coder_ctx = self.coder.get_context()
        self.assertIsNotNone(coder_ctx["latest_resolution"])
        self.assertTrue(0.0 <= coder_ctx["adaptive_gate"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
