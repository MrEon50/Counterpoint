"""
FrictionBus: Cognitive Friction Engine & Polyphonic Tension Bus.
===============================================================

Coordinates the exchange of typed friction vectors F = (F_e, F_v, F_a, F_s)
between all Voices in the Counterpoint Mesh.

Functions:
- Computes the coupled metric norm ||F||_G and collective tension N_col.
- Calculates cross-agent surprise matrix F_s^(i, j) = TV(pi_f^i, pi_s^j).
- Simulates discrete time dynamics of friction with stability enforcement (delta > 3*kappa).
- Computes pairwise opposition distance d_ij = (1 - cos(h_i, h_j)) / 2.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Sequence

from counterpoint.core.types import (
    FrictionVector, AgentState, cosine_similarity, clamp, VectorLike, to_list
)


class FrictionBus:
    """
    Polyphonic friction management bus.
    Maintains the state of tensions across all active Voices in the system.
    """
    def __init__(
        self,
        kappa_N: float = 0.25,          # Coupling parameter in static norm G
        gamma_couple: float = 0.35,      # Weight of inter-agent dispersion in N_col
        delta: float = 0.40,            # Decay coefficient in time dynamics
        kappa: float = 0.10,            # Cross-axis spillover in time dynamics (delta > 3*kappa)
        eta: float = 0.50               # Step size for discrete ODE
    ):
        self.kappa_N = kappa_N
        self.gamma_couple = gamma_couple
        self.delta = delta
        self.kappa = kappa
        self.eta = eta
        
        # Enforce mathematical stability condition: delta > 3*kappa
        if self.delta <= 3.0 * self.kappa:
            # Auto-tune kappa to maintain asymptotic stability
            self.kappa = max(0.01, (self.delta / 3.0) * 0.95)

        # Registry of current state per voice
        self._voice_states: Dict[str, AgentState] = {}
        self._previous_frictions: Dict[str, FrictionVector] = {}
        self._history_n_col: List[float] = []

    def register_step(self, voice_id: str, state: AgentState):
        """Records an agent's submission for the current Bar."""
        if voice_id in self._voice_states:
            self._previous_frictions[voice_id] = self._voice_states[voice_id].friction
        self._voice_states[voice_id] = state

    def get_friction(self, voice_id: str) -> FrictionVector:
        """Retrieves the current friction vector of a voice."""
        if voice_id in self._voice_states:
            return self._voice_states[voice_id].friction
        return FrictionVector()

    def compute_voice_norm(self, voice_id: str) -> float:
        """Calculates normalized tension N_i in [0.0, 1.0] for a specific voice."""
        f = self.get_friction(voice_id)
        return f.compute_norm(self.kappa_N)

    def compute_collective_tension(self) -> float:
        """
        Computes collective tension N_col across all active agents:
          N_col = (1/n_a) * sum(N_i) + gamma_couple * max_{i < j} |N_i - N_j|
        
        Reflects both average team load and specialization divergence (healthy tension).
        """
        if not self._voice_states:
            return 0.0

        norms = [s.friction.compute_norm(self.kappa_N) for s in self._voice_states.values()]
        mean_norm = sum(norms) / len(norms)

        if len(norms) < 2:
            n_col = mean_norm
        else:
            max_dispersion = 0.0
            for i in range(len(norms)):
                for j in range(i + 1, len(norms)):
                    disp = abs(norms[i] - norms[j])
                    if disp > max_dispersion:
                        max_dispersion = disp
            n_col = mean_norm + self.gamma_couple * max_dispersion

        n_col = clamp(n_col)
        self._history_n_col.append(n_col)
        return n_col

    def compute_cross_surprise(self, voice_a: str, voice_b: str) -> float:
        """
        Computes Total Variation distance F_s^(A, B) = TV(pi_f^A, pi_s^B):
          TV(p, q) = 0.5 * sum_k |p_k - q_k|
        Used to measure objective cognitive temperature between two agents.
        """
        state_a = self._voice_states.get(voice_a)
        state_b = self._voice_states.get(voice_b)

        if not state_a or not state_b:
            return 0.0
        if not state_a.policy_logits or not state_b.policy_logits:
            return 0.0

        p = to_list(state_a.policy_logits)
        q = to_list(state_b.policy_logits)
        
        # Softmax normalization if raw logits
        p_dist = self._safe_softmax(p)
        q_dist = self._safe_softmax(q)
        
        # Match dimensions if needed
        min_len = min(len(p_dist), len(q_dist))
        tv = 0.5 * sum(abs(p_dist[k] - q_dist[k]) for k in range(min_len))
        return clamp(tv)

    def compute_pairwise_opposition(self, voice_a: str, voice_b: str) -> float:
        """
        Computes semantic opposition distance f_AB between two agent latent representations:
          f_AB = (1 - cos(h_A, h_B)) / 2  in [0.0, 1.0]
        """
        state_a = self._voice_states.get(voice_a)
        state_b = self._voice_states.get(voice_b)

        if not state_a or not state_b:
            return 0.0
        if not state_a.latent_vector or not state_b.latent_vector:
            # Fallback: estimate from difference in epistemic friction
            return clamp(abs(state_a.friction.epistemic - state_b.friction.epistemic))

        cos_sim = cosine_similarity(state_a.latent_vector, state_b.latent_vector)
        return clamp((1.0 - cos_sim) / 2.0)

    def evolve_friction_dynamics(
        self,
        current_f: FrictionVector,
        external_stimulus: Optional[FrictionVector] = None
    ) -> FrictionVector:
        """
        Applies one discrete step of the coupled ODE:
          F_i(t+1) = clip(F_i(t) + eta * [sigma_i - delta*F_i + kappa*(sum_{j!=i} F_j)*(1 - F_i)])
        Guaranteed asymptotic stability via delta > 3*kappa.
        """
        f = current_f.as_list()
        sigma = external_stimulus.as_list() if external_stimulus else [0.0, 0.0, 0.0, 0.0]
        
        new_f = []
        for i in range(4):
            other_sum = sum(f[j] for j in range(4) if j != i)
            dF_dt = sigma[i] - self.delta * f[i] + self.kappa * other_sum * (1.0 - f[i])
            f_next = clamp(f[i] + self.eta * dF_dt)
            new_f.append(f_next)
            
        return FrictionVector(
            epistemic=new_f[0],
            axiological=new_f[1],
            affective=new_f[2],
            surprise=new_f[3]
        )

    def _safe_softmax(self, logits: List[float]) -> List[float]:
        """Numerically stable softmax."""
        if not logits:
            return []
        max_l = max(logits)
        exps = [math.exp(clamp(x - max_l, min_val=-20.0, max_val=20.0)) for x in logits]
        sum_exps = sum(exps)
        if sum_exps <= 0.0:
            return [1.0 / len(logits)] * len(logits)
        return [e / sum_exps for e in exps]
