"""
LitmusEngine: Leader's Falsification Privilege & Socratic Probe.
================================================================

The Epistemological Leader (voice with highest authority w_i)
earns the right to generate one falsification test (Litmus Probe)
that the rival's proposal must pass before acceptance.

Instead of persuading through words, the Leader says:
  "You claim your solution is correct? Let's verify it on this edge case."

If the rival's proposal fails the Leader's Litmus Probe (confirmed by
the compiler/runtime), that proposal is immediately eliminated —
no grey compromise allowed.

This mechanism prevents:
- Argumentum ad temperantiam (false middle ground).
- Hallucinatory confidence overriding tested correctness.
- Social pressure / verbose persuasion bypassing logical truth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any, Tuple


# ============================================================
# Litmus Probe Result
# ============================================================

@dataclass
class LitmusResult:
    """Outcome of a Leader's Litmus Probe on a rival's proposal."""
    probe_author: str                     # Voice ID of the Leader who created the probe
    target_voice: str                     # Voice ID of the rival being tested
    probe_description: str                # Human-readable description of the test
    probe_passed: bool                    # True if the rival's proposal passed the test
    rival_content_tested: str             # The content that was tested
    message: str = ""                     # Explanation of the result
    probe_fn_name: str = ""               # Name/label of the probe function


# ============================================================
# Litmus Engine
# ============================================================

class LitmusEngine:
    """
    Manages the Leader's Falsification Privilege.
    
    Workflow in each Bar (tick):
    1. EpistemicLedger identifies the Leader (highest authority voice).
    2. Leader generates a Litmus Probe: a callable predicate that tests
       a specific edge case, invariant, or counterexample.
    3. The rival's synthesized proposal O' is subjected to the Litmus Probe.
    4. If O' fails the probe, the synthesis is rejected immediately.
    5. If O' passes, the synthesis proceeds to the standard anchor gate.
    
    The Litmus Probe is an ADDITIONAL verification layer on top of the
    existing external programmatic anchor (v_ext). Together they form
    a double-gate verification system:
      Gate 1: Litmus Probe (Leader's edge case) → must pass
      Gate 2: External Anchor (v_ext)           → must pass
    Both must return True for synthesis acceptance.
    """

    def __init__(self):
        self.probe_history: List[LitmusResult] = []
        self._registered_probes: Dict[str, Callable[[str], bool]] = {}

    def register_probe(
        self,
        leader_voice_id: str,
        probe_fn: Callable[[str], bool],
        description: str = "",
        probe_name: str = ""
    ):
        """
        Registers a Litmus Probe created by the current Leader.
        
        Args:
            leader_voice_id: The Leader who authors this test.
            probe_fn: A callable(content) -> bool that tests whether
                      the given content/solution passes the edge case.
            description: Human-readable explanation of what the probe tests.
            probe_name: Optional short name for the probe function.
        """
        self._registered_probes[leader_voice_id] = probe_fn
        # Store metadata for audit trail
        self._last_description = description
        self._last_probe_name = probe_name or f"litmus_{leader_voice_id}"
        self._last_leader = leader_voice_id

    def has_active_probe(self, leader_voice_id: str) -> bool:
        """Returns True if the specified leader has a registered Litmus Probe."""
        return leader_voice_id in self._registered_probes

    def execute_probe(
        self,
        leader_voice_id: str,
        target_voice_id: str,
        content_to_test: str
    ) -> LitmusResult:
        """
        Executes the Leader's Litmus Probe against a rival's proposal.
        
        Args:
            leader_voice_id: The Leader who authored the probe.
            target_voice_id: The rival whose content is being tested.
            content_to_test: The synthesized content O' to verify.
        
        Returns:
            LitmusResult with pass/fail verdict.
        """
        probe_fn = self._registered_probes.get(leader_voice_id)
        if probe_fn is None:
            # No probe registered — auto-pass (Leader chose not to challenge)
            result = LitmusResult(
                probe_author=leader_voice_id,
                target_voice=target_voice_id,
                probe_description="No probe registered",
                probe_passed=True,
                rival_content_tested=content_to_test,
                message="Leader did not register a Litmus Probe — auto-pass"
            )
            self.probe_history.append(result)
            return result

        # Execute the probe with safety wrapper
        try:
            passed = bool(probe_fn(content_to_test))
        except Exception as e:
            # Probe execution error — treat as failure (fail-safe)
            passed = False
            result = LitmusResult(
                probe_author=leader_voice_id,
                target_voice=target_voice_id,
                probe_description=getattr(self, '_last_description', ''),
                probe_passed=False,
                rival_content_tested=content_to_test,
                message=f"Litmus Probe raised exception: {str(e)}",
                probe_fn_name=getattr(self, '_last_probe_name', '')
            )
            self.probe_history.append(result)
            return result

        description = getattr(self, '_last_description', '')
        probe_name = getattr(self, '_last_probe_name', '')

        if passed:
            msg = f"Litmus Probe PASSED: rival's proposal survived Leader's edge case"
        else:
            msg = f"Litmus Probe FAILED: rival's proposal eliminated by Leader's falsification test"

        result = LitmusResult(
            probe_author=leader_voice_id,
            target_voice=target_voice_id,
            probe_description=description,
            probe_passed=passed,
            rival_content_tested=content_to_test,
            message=msg,
            probe_fn_name=probe_name
        )
        self.probe_history.append(result)
        return result

    def clear_probes(self):
        """Clears all registered probes (typically done between Bars)."""
        self._registered_probes.clear()

    def get_history(self) -> List[LitmusResult]:
        """Returns the full Litmus Probe audit trail."""
        return list(self.probe_history)

    def get_stats(self) -> Dict[str, Any]:
        """Returns summary statistics of Litmus Probe outcomes."""
        if not self.probe_history:
            return {"total_probes": 0, "pass_rate": 0.0, "eliminations": 0}

        total = len(self.probe_history)
        passed = sum(1 for r in self.probe_history if r.probe_passed)
        return {
            "total_probes": total,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "eliminations": total - passed,
            "latest_probe": self.probe_history[-1].message if self.probe_history else ""
        }
