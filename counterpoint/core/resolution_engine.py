"""
ResolutionEngine: Dialectical Synthesis Gate & Axiom of Recursion R(A, B) -> O'.
==============================================================================

The core emergent generator of Counterpoint:
When two agents experience cognitive opposition (f_AB >= 0.3), instead of
allowing an endless natural language argumentation loop, the ResolutionEngine:
1. Computes the Beauty Index B(A, B) to prioritize productive conflicts.
2. Invokes synthesis: R(A, B) -> O' (generating a higher abstraction).
3. Gates ratification through an external Programmatic Anchor (v_ext in {0, 1}).
4. Evaluates synthesis gain: Delta_syn = N_before - N_after - mu * complexity.
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Dict, List, Any, Tuple

from counterpoint.core.types import (
    SynthesisResult, AgentState, FrictionVector, clamp, to_list, vector_norm
)


class ResolutionEngine:
    """
    Synthesis Gate and Dialectical Resolver.
    Enforces the Axiom of Recursion: A validated relation between two states
    collapses tension and becomes a new object O' on the collective score.
    """
    def __init__(
        self,
        opposition_threshold: float = 0.30,   # Minimum f_AB to trigger synthesis
        complexity_penalty: float = 0.05,     # mu: penalty for unnecessary complexity
        synthesizer_fn: Optional[Callable[[str, str, Dict[str, Any]], str]] = None
    ):
        self.opposition_threshold = opposition_threshold
        self.complexity_penalty = complexity_penalty
        self.synthesizer_fn = synthesizer_fn or self._default_synthesizer
        self.resolution_history: List[SynthesisResult] = []

    def compute_beauty_index(
        self,
        f_AB: float,
        depth_a: float = 1.0,
        depth_b: float = 1.0,
        n_active: int = 2
    ) -> float:
        """
        Calculates the Beauty Index B(A, B) (from CLA / Mind Alphabet):
          B(A, B) = [ 0.5 * (depth_A + depth_B) ] / [ f_AB + 0.05 * n_act + 0.10 ]
        
        Rewards deep structural conflicts while penalizing overcrowded or shallow friction.
        """
        numerator = 0.5 * (depth_a + depth_b)
        denominator = f_AB + 0.05 * float(n_active) + 0.10
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    def attempt_resolution(
        self,
        voice_a: str,
        state_a: AgentState,
        voice_b: str,
        state_b: AgentState,
        opposition_score: float,
        collective_tension_before: float,
        verifier_fn: Optional[Callable[[str], bool]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> SynthesisResult:
        """
        Evaluates whether opposition between voice_a and voice_b triggers a resolution.
        If triggered, synthesizes O' and subjects it to programmatic verification.
        """
        context = context or {}
        n_act = context.get("n_active", 2)
        depth_a = context.get("depth_a", 1.0)
        depth_b = context.get("depth_b", 1.0)
        
        beauty = self.compute_beauty_index(opposition_score, depth_a, depth_b, n_act)

        # Check activation threshold
        if opposition_score < self.opposition_threshold:
            return SynthesisResult(
                triggered=False,
                voice_a=voice_a,
                voice_b=voice_b,
                opposition_score=opposition_score,
                beauty_index=beauty,
                message=f"Opposition {opposition_score:.3f} below threshold {self.opposition_threshold}"
            )

        # 1. Synthesize candidate content O'
        try:
            o_prime_content = self.synthesizer_fn(state_a.content, state_b.content, context)
        except Exception as e:
            o_prime_content = f"SYNTHESIS_ERROR: {str(e)}"

        # 2. Synthesize candidate vector representation
        o_prime_vec = self._synthesize_vectors(state_a.latent_vector, state_b.latent_vector)

        # 3. External Programmatic Anchor (Cadenza Gate)
        v_ext = 1
        if verifier_fn is not None:
            try:
                passed = verifier_fn(o_prime_content)
                v_ext = 1 if passed else 0
            except Exception:
                v_ext = 0

        # 4. Compute complexity and synthesis gain Delta_syn
        # Measure complexity as relative length expansion or vector entropy
        len_orig = max(1, len(state_a.content) + len(state_b.content))
        len_synth = len(o_prime_content)
        complexity_measure = max(0.0, float(len_synth) / float(len_orig) - 0.5)

        # Post-synthesis tension drops when external anchor passes
        tension_after = collective_tension_before * 0.4 if v_ext == 1 else collective_tension_before
        delta_syn = (collective_tension_before - tension_after) - (self.complexity_penalty * complexity_measure)

        # 5. Strict acceptance criterion: Delta_syn > 0 AND v_ext == 1
        accepted = (delta_syn > 0.0 and v_ext == 1)

        result = SynthesisResult(
            triggered=True,
            voice_a=voice_a,
            voice_b=voice_b,
            opposition_score=round(opposition_score, 4),
            beauty_index=round(beauty, 4),
            synthesized_content=o_prime_content,
            synthesized_vector=o_prime_vec,
            delta_syn=round(delta_syn, 4),
            external_verdict=v_ext,
            accepted=accepted,
            message="Synthesis ratified by programmatic anchor" if accepted else "Synthesis rejected (anchor failed or gain <= 0)"
        )

        self.resolution_history.append(result)
        return result

    def _synthesize_vectors(
        self,
        vec_a: Optional[List[float]],
        vec_b: Optional[List[float]]
    ) -> Optional[List[float]]:
        """
        Differentiable neural/vector synthesis:
          O' = normalize(0.5 * (vec_a + vec_b) + interaction)
        """
        if not vec_a or not vec_b:
            return None
        min_len = min(len(vec_a), len(vec_b))
        if min_len == 0:
            return None

        # Mean representation + non-linear interaction
        o_prime = []
        for i in range(min_len):
            m = 0.5 * (vec_a[i] + vec_b[i])
            diff = abs(vec_a[i] - vec_b[i])
            o_prime.append(math.tanh(m + 0.1 * diff))

        # Normalize
        norm = vector_norm(o_prime)
        if norm > 1e-9:
            return [x / norm for x in o_prime]
        return o_prime

    def _default_synthesizer(self, content_a: str, content_b: str, context: Dict[str, Any]) -> str:
        """
        Default structured rule-based dialectical synthesizer.
        Extracts complementary blocks and merges them.
        """
        header = f"# Resolved Synthesis [R({context.get('voice_a', 'A')}, {context.get('voice_b', 'B')}) -> O']\n"
        # If one is code and the other is a critique, refine code with critique
        if "```" in content_a or "def " in content_a:
            return f"{header}\n// Synthesized Code Resolution:\n{content_a}\n// Addressed Criticisms:\n{content_b}"
        elif "```" in content_b or "def " in content_b:
            return f"{header}\n// Synthesized Code Resolution:\n{content_b}\n// Addressed Criticisms:\n{content_a}"
        return f"{header}\n--- Perspective A ---\n{content_a}\n--- Perspective B ---\n{content_b}\n--- Integrated Resolution ---\nMerged proposition reconciling A and B."
