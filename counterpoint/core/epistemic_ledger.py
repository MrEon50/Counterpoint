"""
EpistemicLedger: Dynamic Authority & Anti-Compromise Truth Tracker.
===================================================================

Solves the "Fallacy of the Golden Mean" (Argumentum ad temperantiam):
When agent A says "Earth is round" (truth) and agent B says "Earth is flat"
(false), naive systems average to "Earth is a hemisphere" (absurd compromise).

EpistemicLedger ensures:
1. Ground-Truth Probing: calibrate each voice's accuracy on known-answer tasks.
2. Asymmetric Truth Gravity: weight synthesis toward the more credible voice
   instead of naive (A+B)/2 averaging.
3. Overconfidence Penalty: punish voices that are highly confident but wrong.
4. Epistemic Quarantine: demote persistently inaccurate voices to Listener mode.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import deque

from counterpoint.core.types import clamp


# ============================================================
# Calibration Probe Record
# ============================================================

@dataclass
class CalibrationRecord:
    """Result of a single ground-truth calibration probe."""
    voice_id: str
    question_id: str
    expected_answer: str
    voice_answer: str
    correct: bool
    confidence: float            # Self-reported confidence in [0, 1]
    overconfidence_penalty: float # |confidence - accuracy| when wrong


# ============================================================
# Epistemic Ledger
# ============================================================

class EpistemicLedger:
    """
    Tracks per-voice epistemic authority w_i in [0.0, 1.0].
    
    Authority is earned through:
      - Passing ground-truth calibration probes (accuracy bonus).
      - Contributing to accepted syntheses (synthesis bonus).
    
    Authority is lost through:
      - Failing calibration probes (accuracy penalty).
      - Overconfidence: being wrong with high self-reported confidence
        (penalty = beta * |confidence - actual_accuracy|).
      - Persistent low authority triggers Epistemic Quarantine
        (voice loses decision weight, becomes Listener).
    
    Asymmetric Truth Gravity:
      When two voices disagree, instead of (A+B)/2, the resolution
      uses weighted projection biased toward the more credible voice:
        direction = w_A * A + w_B * B  (where w_A + w_B = 1, normalized)
      If authority gap > dominance_threshold, the weaker voice's
      hypothesis is rejected entirely (no grey compromise).
    """

    def __init__(
        self,
        initial_authority: float = 0.50,
        accuracy_bonus: float = 0.08,
        accuracy_penalty: float = 0.12,
        overconfidence_beta: float = 0.15,
        synthesis_bonus: float = 0.05,
        synthesis_penalty: float = 0.07,
        quarantine_threshold: float = 0.15,
        dominance_threshold: float = 0.40,
        calibration_window: int = 20
    ):
        self.initial_authority = initial_authority
        self.accuracy_bonus = accuracy_bonus
        self.accuracy_penalty = accuracy_penalty
        self.overconfidence_beta = overconfidence_beta
        self.synthesis_bonus = synthesis_bonus
        self.synthesis_penalty = synthesis_penalty
        self.quarantine_threshold = quarantine_threshold
        self.dominance_threshold = dominance_threshold
        self.calibration_window = calibration_window

        # Per-voice state
        self._authority: Dict[str, float] = {}
        self._calibration_history: Dict[str, deque] = {}
        self._quarantined: Dict[str, bool] = {}
        self._quarantine_reasons: Dict[str, str] = {}

    def register_voice(self, voice_id: str, initial_w: Optional[float] = None):
        """Registers a voice with initial epistemic authority."""
        w = initial_w if initial_w is not None else self.initial_authority
        self._authority[voice_id] = clamp(w)
        self._calibration_history[voice_id] = deque(maxlen=self.calibration_window)
        self._quarantined[voice_id] = False
        self._quarantine_reasons[voice_id] = ""

    def get_authority(self, voice_id: str) -> float:
        """Returns the current epistemic weight w_i in [0, 1] for a voice."""
        return self._authority.get(voice_id, self.initial_authority)

    def is_quarantined(self, voice_id: str) -> bool:
        """Returns True if this voice has been demoted to Listener."""
        return self._quarantined.get(voice_id, False)

    def get_quarantine_reason(self, voice_id: str) -> str:
        """Returns the reason for quarantine, or empty string."""
        return self._quarantine_reasons.get(voice_id, "")

    # --------------------------------------------------------
    # Ground-Truth Calibration
    # --------------------------------------------------------

    def run_calibration_probe(
        self,
        voice_id: str,
        question_id: str,
        expected_answer: str,
        voice_answer: str,
        confidence: float = 0.5
    ) -> CalibrationRecord:
        """
        Evaluates a single ground-truth probe for a voice.
        
        Args:
            voice_id: The voice being tested.
            question_id: Unique ID of the calibration question.
            expected_answer: The known correct answer.
            voice_answer: The voice's submitted answer.
            confidence: Voice's self-reported confidence in [0, 1].
        
        Returns:
            CalibrationRecord with accuracy and penalty details.
        """
        if voice_id not in self._authority:
            self.register_voice(voice_id)

        confidence = clamp(confidence)
        correct = (voice_answer.strip().lower() == expected_answer.strip().lower())

        # Compute overconfidence penalty
        if correct:
            oc_penalty = 0.0
            # Reward: accuracy bonus
            self._authority[voice_id] = clamp(
                self._authority[voice_id] + self.accuracy_bonus
            )
        else:
            # Penalty: base accuracy loss + overconfidence surcharge
            oc_penalty = self.overconfidence_beta * confidence
            total_penalty = self.accuracy_penalty + oc_penalty
            self._authority[voice_id] = clamp(
                self._authority[voice_id] - total_penalty
            )

        record = CalibrationRecord(
            voice_id=voice_id,
            question_id=question_id,
            expected_answer=expected_answer,
            voice_answer=voice_answer,
            correct=correct,
            confidence=confidence,
            overconfidence_penalty=round(oc_penalty, 4)
        )
        self._calibration_history[voice_id].append(record)

        # Check quarantine condition
        self._check_quarantine(voice_id)
        return record

    def run_calibration_suite(
        self,
        voice_id: str,
        probes: List[Dict[str, Any]],
        answer_fn: Callable[[str], Tuple[str, float]]
    ) -> List[CalibrationRecord]:
        """
        Runs a batch of calibration probes for a single voice.
        
        Args:
            voice_id: The voice being calibrated.
            probes: List of dicts with keys 'question_id', 'question', 'expected_answer'.
            answer_fn: Callable(question) -> (answer, confidence).
        
        Returns:
            List of CalibrationRecords.
        """
        results = []
        for probe in probes:
            answer, conf = answer_fn(probe["question"])
            record = self.run_calibration_probe(
                voice_id=voice_id,
                question_id=probe["question_id"],
                expected_answer=probe["expected_answer"],
                voice_answer=answer,
                confidence=conf
            )
            results.append(record)
        return results

    # --------------------------------------------------------
    # Synthesis Feedback
    # --------------------------------------------------------

    def record_synthesis_outcome(
        self,
        voice_id: str,
        synthesis_accepted: bool
    ):
        """
        Updates authority based on whether this voice's contribution
        to a dialectical synthesis was accepted or rejected.
        """
        if voice_id not in self._authority:
            self.register_voice(voice_id)

        if synthesis_accepted:
            self._authority[voice_id] = clamp(
                self._authority[voice_id] + self.synthesis_bonus
            )
        else:
            self._authority[voice_id] = clamp(
                self._authority[voice_id] - self.synthesis_penalty
            )

        self._check_quarantine(voice_id)

    # --------------------------------------------------------
    # Asymmetric Truth Gravity
    # --------------------------------------------------------

    def compute_asymmetric_weights(
        self,
        voice_a: str,
        voice_b: str
    ) -> Tuple[float, float, bool]:
        """
        Computes asymmetric synthesis weights for two opposing voices.
        
        Returns:
            (w_A_norm, w_B_norm, compromise_rejected):
            - w_A_norm, w_B_norm: Normalized weights summing to 1.0,
              biased toward the more credible voice.
            - compromise_rejected: True if authority gap exceeds
              dominance_threshold (weaker voice eliminated entirely).
        """
        w_a = self.get_authority(voice_a)
        w_b = self.get_authority(voice_b)

        # If one is quarantined, the other dominates completely
        if self.is_quarantined(voice_a) and not self.is_quarantined(voice_b):
            return 0.0, 1.0, True
        if self.is_quarantined(voice_b) and not self.is_quarantined(voice_a):
            return 1.0, 0.0, True

        gap = abs(w_a - w_b)
        compromise_rejected = gap > self.dominance_threshold

        if compromise_rejected:
            # Winner takes all — no grey compromise
            if w_a >= w_b:
                return 1.0, 0.0, True
            else:
                return 0.0, 1.0, True

        # Proportional weighting (soft asymmetry)
        total = w_a + w_b
        if total < 1e-9:
            return 0.5, 0.5, False
        return round(w_a / total, 4), round(w_b / total, 4), False

    def get_leader(self, voice_ids: List[str]) -> Optional[str]:
        """
        Returns the voice_id with the highest epistemic authority
        among the given voices. This voice earns the Litmus Test privilege.
        """
        if not voice_ids:
            return None

        best_id = None
        best_w = -1.0
        for vid in voice_ids:
            w = self.get_authority(vid)
            if w > best_w and not self.is_quarantined(vid):
                best_w = w
                best_id = vid
        return best_id

    # --------------------------------------------------------
    # Quarantine Logic
    # --------------------------------------------------------

    def _check_quarantine(self, voice_id: str):
        """Demotes a voice to Listener if authority drops below threshold."""
        w = self._authority.get(voice_id, self.initial_authority)
        if w <= self.quarantine_threshold:
            self._quarantined[voice_id] = True
            self._quarantine_reasons[voice_id] = (
                f"Epistemic Quarantine: authority {w:.3f} <= "
                f"threshold {self.quarantine_threshold}"
            )
        else:
            # Allow recovery if authority climbs back above threshold
            if self._quarantined.get(voice_id, False):
                self._quarantined[voice_id] = False
                self._quarantine_reasons[voice_id] = ""

    def lift_quarantine(self, voice_id: str):
        """Manually lifts quarantine for a voice (e.g., after re-calibration)."""
        self._quarantined[voice_id] = False
        self._quarantine_reasons[voice_id] = ""

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def get_accuracy_rate(self, voice_id: str) -> float:
        """Returns the recent calibration accuracy rate in [0, 1]."""
        history = self._calibration_history.get(voice_id)
        if not history:
            return 0.0
        correct_count = sum(1 for r in history if r.correct)
        return correct_count / len(history)

    def get_status(self) -> Dict[str, Any]:
        """Returns a summary of all voice authorities and quarantine status."""
        return {
            vid: {
                "authority": round(self._authority[vid], 4),
                "quarantined": self._quarantined.get(vid, False),
                "accuracy_rate": round(self.get_accuracy_rate(vid), 4),
                "calibration_probes": len(self._calibration_history.get(vid, [])),
                "quarantine_reason": self._quarantine_reasons.get(vid, "")
            }
            for vid in self._authority
        }
