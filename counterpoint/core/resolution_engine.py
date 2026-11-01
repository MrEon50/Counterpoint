"""
ResolutionEngine: Dialectical Synthesis Gate & Axiom of Recursion R(A, B) -> O'.
==============================================================================

The core emergent generator of Counterpoint:
When two agents experience cognitive opposition (f_AB >= 0.3), instead of
allowing an endless natural language argumentation loop, the ResolutionEngine:
1. Computes the Beauty Index B(A, B) to prioritize productive conflicts.
2. Applies Asymmetric Truth Gravity via EpistemicLedger weights.
3. Invokes synthesis: R(A, B) -> O' (generating a higher abstraction).
4. Subjects O' to the Leader's Litmus Probe (LitmusEngine falsification gate).
5. Gates ratification through an external Programmatic Anchor (v_ext in {0, 1}).
6. Evaluates synthesis gain: Delta_syn = N_before - N_after - mu * complexity.
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Dict, List, Any, Tuple, TYPE_CHECKING

from counterpoint.core.types import (
    SynthesisResult, AgentState, FrictionVector, clamp, to_list, vector_norm
)

if TYPE_CHECKING:
    from counterpoint.core.epistemic_ledger import EpistemicLedger
    from counterpoint.core.litmus_engine import LitmusEngine


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
        synthesizer_fn: Optional[Callable[[str, str, Dict[str, Any]], str]] = None,
        ledger: Optional["EpistemicLedger"] = None,
        litmus_engine: Optional["LitmusEngine"] = None
    ):
        self.opposition_threshold = opposition_threshold
        self.complexity_penalty = complexity_penalty
        self.synthesizer_fn = synthesizer_fn or self._default_synthesizer
        self.ledger = ledger
        self.litmus_engine = litmus_engine
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
        If triggered, synthesizes O' and subjects it to:
          Gate 1: Asymmetric Truth Gravity (EpistemicLedger weights).
          Gate 2: Leader's Litmus Probe (LitmusEngine falsification test).
          Gate 3: External Programmatic Anchor (v_ext in {0, 1}).
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

        # --- Gate 1: Asymmetric Truth Gravity ---
        asymmetric_weights: Dict[str, float] = {}
        compromise_rejected = False

        if self.ledger is not None:
            w_a, w_b, compromise_rejected = self.ledger.compute_asymmetric_weights(
                voice_a, voice_b
            )
            asymmetric_weights = {voice_a: w_a, voice_b: w_b}

            if compromise_rejected:
                # Winner takes all — no grey averaging
                if w_a > w_b:
                    winner_state, loser_voice = state_a, voice_b
                else:
                    winner_state, loser_voice = state_b, voice_a

                result = SynthesisResult(
                    triggered=True,
                    voice_a=voice_a,
                    voice_b=voice_b,
                    opposition_score=round(opposition_score, 4),
                    beauty_index=round(beauty, 4),
                    synthesized_content=winner_state.content,
                    synthesized_vector=winner_state.latent_vector,
                    delta_syn=round(collective_tension_before * 0.5, 4),
                    external_verdict=1,
                    accepted=True,
                    message=f"Compromise rejected: authority gap too large, "
                            f"{loser_voice} eliminated by Asymmetric Truth Gravity",
                    asymmetric_weights=asymmetric_weights,
                    compromise_rejected=True
                )

                # Update ledger: winner rewarded, loser penalized
                if w_a > w_b:
                    self.ledger.record_synthesis_outcome(voice_a, True)
                    self.ledger.record_synthesis_outcome(voice_b, False)
                else:
                    self.ledger.record_synthesis_outcome(voice_b, True)
                    self.ledger.record_synthesis_outcome(voice_a, False)

                self.resolution_history.append(result)
                return result
        else:
            # No ledger — equal weights (backward compatible)
            asymmetric_weights = {voice_a: 0.5, voice_b: 0.5}

        # --- Synthesize candidate content O' ---
        # Pass asymmetric weights into context so synthesizer can use them
        synth_context = dict(context)
        synth_context["asymmetric_weights"] = asymmetric_weights
        try:
            o_prime_content = self.synthesizer_fn(state_a.content, state_b.content, synth_context)
        except Exception as e:
            o_prime_content = f"SYNTHESIS_ERROR: {str(e)}"

        # Synthesize candidate vector representation (weight-biased)
        w_a_val = asymmetric_weights.get(voice_a, 0.5)
        w_b_val = asymmetric_weights.get(voice_b, 0.5)
        o_prime_vec = self._synthesize_vectors(
            state_a.latent_vector, state_b.latent_vector,
            weight_a=w_a_val, weight_b=w_b_val
        )

        # --- Gate 2: Leader's Litmus Probe ---
        litmus_passed = True
        litmus_verdict: Optional[bool] = None

        if self.litmus_engine is not None and self.ledger is not None:
            leader = self.ledger.get_leader([voice_a, voice_b])
            if leader and self.litmus_engine.has_active_probe(leader):
                target = voice_b if leader == voice_a else voice_a
                litmus_result = self.litmus_engine.execute_probe(
                    leader_voice_id=leader,
                    target_voice_id=target,
                    content_to_test=o_prime_content
                )
                litmus_passed = litmus_result.probe_passed
                litmus_verdict = litmus_result.probe_passed

        # --- Gate 3: External Programmatic Anchor (Cadenza Gate) ---
        v_ext = 1
        if verifier_fn is not None:
            try:
                passed = verifier_fn(o_prime_content)
                v_ext = 1 if passed else 0
            except Exception:
                v_ext = 0

        # --- Compute complexity and synthesis gain Delta_syn ---
        len_orig = max(1, len(state_a.content) + len(state_b.content))
        len_synth = len(o_prime_content)
        complexity_measure = max(0.0, float(len_synth) / float(len_orig) - 0.5)

        # Post-synthesis tension drops when all gates pass
        all_gates_pass = (v_ext == 1 and litmus_passed)
        tension_after = collective_tension_before * 0.4 if all_gates_pass else collective_tension_before
        delta_syn = (collective_tension_before - tension_after) - (self.complexity_penalty * complexity_measure)

        # --- Strict acceptance: Delta_syn > 0 AND v_ext == 1 AND litmus passed ---
        accepted = (delta_syn > 0.0 and v_ext == 1 and litmus_passed)

        # Build acceptance message
        if accepted:
            message = "Synthesis ratified (anchor + litmus passed)"
        elif not litmus_passed:
            message = "Synthesis rejected by Leader's Litmus Probe"
        elif v_ext == 0:
            message = "Synthesis rejected (programmatic anchor failed)"
        else:
            message = "Synthesis rejected (gain <= 0)"

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
            message=message,
            asymmetric_weights=asymmetric_weights,
            litmus_verdict=litmus_verdict,
            compromise_rejected=False
        )

        # Update ledger with synthesis outcome
        if self.ledger is not None:
            self.ledger.record_synthesis_outcome(voice_a, accepted)
            self.ledger.record_synthesis_outcome(voice_b, accepted)

        self.resolution_history.append(result)
        return result

    def _synthesize_vectors(
        self,
        vec_a: Optional[List[float]],
        vec_b: Optional[List[float]],
        weight_a: float = 0.5,
        weight_b: float = 0.5
    ) -> Optional[List[float]]:
        """
        Asymmetric vector synthesis:
          O' = normalize(w_A * vec_a + w_B * vec_b + interaction)
        Uses epistemic weights instead of naive 0.5/0.5 averaging.
        """
        if not vec_a or not vec_b:
            return None
        min_len = min(len(vec_a), len(vec_b))
        if min_len == 0:
            return None

        # Weighted representation + non-linear interaction
        o_prime = []
        for i in range(min_len):
            m = weight_a * vec_a[i] + weight_b * vec_b[i]
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
