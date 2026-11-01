"""
EquilibriumGovernor: Polarized Harmony & Anti-Goodhart RSI Guard.
================================================================

Adapts DRM (Dynamic Rule Matrix) Equilibrium principles to multi-agent meshes:
1. System Harmony Phi in [-1.0, 1.0]: Balances expansion (+P, innovation, bold code)
   against protection (-P, unit tests, memory limits, safety assertions).
2. Stagnation Detection: Applies controlled perturbation shock to shake teams
   out of consensus paralysis (groupthink).
3. Goodhart Gap Guard: Gamma = E[1 - N] - E[success]. Prevents reward hacking where
   agents lower internal friction without solving actual user problems.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from counterpoint.core.types import clamp


@dataclass
class RuleDirective:
    """A governed rule or guideline inside the multi-agent mesh."""
    id: str
    name: str
    polarity: float             # in [-1.0, 1.0]: +1 for expansion/creativity, -1 for protection/safety
    weight: float = 1.0         # Current dynamic weight in [0.01, 100.0]
    priority: int = 1           # Priority level (1 to 5)
    magnitude: float = 1.0      # Inherent force magnitude
    quarantined: bool = False
    quarantine_reason: str = ""


class EquilibriumGovernor:
    """
    Regulates the equilibrium between creative exploration and rigorous verification.
    """
    def __init__(
        self,
        target_harmony: float = 0.0,
        tolerance: float = 0.30,
        stagnation_variance_threshold: float = 0.005,
        window_size: int = 15
    ):
        self.target_harmony = target_harmony
        self.tolerance = tolerance
        self.stagnation_variance_threshold = stagnation_variance_threshold
        self.window_size = window_size

        self.rules: Dict[str, RuleDirective] = {}
        self._phi_history: deque = deque(maxlen=window_size)
        self._success_history: deque = deque(maxlen=50)
        self._tension_history: deque = deque(maxlen=50)

        # Register default mesh directives
        self._init_default_rules()

    def _init_default_rules(self):
        """Initializes default expansion and protection directives."""
        self.add_rule(RuleDirective(
            id="creative_exploration",
            name="Explore Novel Solutions & Synthesis",
            polarity=0.8,
            priority=3,
            magnitude=1.2
        ))
        self.add_rule(RuleDirective(
            id="unit_test_safety",
            name="Strict Programmatic Anchor & Test Pass",
            polarity=-0.9,
            priority=5,
            magnitude=1.5
        ))
        self.add_rule(RuleDirective(
            id="token_budget_guard",
            name="Token Economy & Conciseness",
            polarity=-0.6,
            priority=2,
            magnitude=1.0
        ))

    def add_rule(self, rule: RuleDirective):
        """Registers a governed rule directive."""
        self.rules[rule.id] = rule

    def calculate_system_harmony(self) -> float:
        """
        Computes system harmony Phi in [-1.0, 1.0]:
          Phi = sum(polarity_i * effective_weight_i) / sum(effective_weight_i)
        """
        active_rules = [r for r in self.rules.values() if not r.quarantined]
        if not active_rules:
            return 0.0

        total_wp = 0.0
        total_ew = 0.0
        for r in active_rules:
            p_weight = max(1.0, float(r.priority + 1))
            ew = max(1e-6, r.weight * p_weight * r.magnitude)
            total_wp += r.polarity * ew
            total_ew += ew

        if total_ew <= 1e-6:
            return 0.0

        phi = clamp(total_wp / total_ew, min_val=-1.0, max_val=1.0)
        self._phi_history.append(phi)
        return phi

    def balance_weights(self) -> Dict[str, float]:
        """
        Dynamically adjusts rule weights when harmony drifts beyond tolerance:
          - If Phi > target + tol (too risky) -> increase safety, reduce expansion.
          - If Phi < target - tol (too passive) -> increase expansion, reduce safety.
        Returns adjusted weight map.
        """
        phi = self.calculate_system_harmony()
        if abs(phi - self.target_harmony) <= self.tolerance:
            return {r.id: r.weight for r in self.rules.values()}

        for r in self.rules.values():
            if r.quarantined:
                continue
            if phi > self.target_harmony:
                # System too expansive/risky -> boost protection, dampen expansion
                if r.polarity < 0:
                    r.weight = min(100.0, r.weight * 1.20)
                elif r.polarity > 0:
                    r.weight = max(0.01, r.weight * 0.85)
            else:
                # System too conservative/stagnant -> boost expansion, dampen protection
                if r.polarity > 0:
                    r.weight = min(100.0, r.weight * 1.20)
                elif r.polarity < 0:
                    r.weight = max(0.01, r.weight * 0.85)

        return {r.id: r.weight for r in self.rules.values()}

    def detect_stagnation_and_perturb(self) -> Tuple[bool, Optional[str]]:
        """
        Detects consensus stagnation (groupthink / flatlined tension).
        If variance of Phi is below threshold over the window, applies a perturbation shock.
        """
        if len(self._phi_history) < self.window_size:
            return False, None

        mean_phi = sum(self._phi_history) / len(self._phi_history)
        variance = sum((p - mean_phi) ** 2 for p in self._phi_history) / len(self._phi_history)

        if variance < self.stagnation_variance_threshold:
            # Apply symmetric perturbation shock to shake the system
            shock_dir = 1.0 if mean_phi <= 0.0 else -1.0
            for r in self.rules.values():
                if r.polarity * shock_dir > 0:
                    r.weight = min(100.0, r.weight * 1.4)
            return True, f"Perturbation shock applied (variance {variance:.5f} < {self.stagnation_variance_threshold})"

        return False, None

    def record_step_outcome(self, collective_tension: float, success: bool):
        """Records an execution outcome to track the Goodhart gap."""
        self._tension_history.append(collective_tension)
        self._success_history.append(1.0 if success else 0.0)

    def check_goodhart_gap(self) -> Tuple[float, bool]:
        """
        Computes the Goodhart divergence metric:
          Gamma = E[1 - N] - E[success]
        
        When Gamma > 0: the system has low internal friction but fails tasks
        (reward hacking / pretending to agree). Returns (Gamma, is_hacking).
        """
        if len(self._success_history) < 5:
            return 0.0, False

        mean_n = sum(self._tension_history) / len(self._tension_history)
        mean_succ = sum(self._success_history) / len(self._success_history)

        # High (1 - N) means low friction (apparent harmony)
        perceived_harmony = 1.0 - mean_n
        gamma = perceived_harmony - mean_succ

        is_hacking = (gamma > 0.35 and mean_succ < 0.50)
        if is_hacking:
            # Quarantine the most expansive rule to force rigor
            for r in self.rules.values():
                if r.polarity > 0.5 and not r.quarantined:
                    r.quarantined = True
                    r.quarantine_reason = f"Quarantined by Goodhart Guard (Gamma={gamma:.3f})"
                    break

        return gamma, is_hacking
