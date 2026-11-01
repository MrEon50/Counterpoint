"""
ScoreMemory: GCL-based Causal Blackboard O(n) & Adaptive Attention Gate.
========================================================================

Implements the "Score" (global musical score / collective canvas):
1. Causal Cumulative Pooling O(n): Aggregates hidden state vectors from all Voices
   with strictly O(1) memory per step.
2. Global Schema Generation: Compresses multi-agent context into schema dimension S.
3. Adaptive Gate (g_i in [0, 1]): Each voice autonomously receives an attention weight
   indicating how much to attend to collective consensus vs local focus.
4. Schema Reflection: Enriches agent representations under the influence of the global schema.
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Optional, Tuple, Union

from counterpoint.core.types import (
    VectorLike, to_list, dot_product, vector_norm, clamp, HAS_NUMPY
)

if HAS_NUMPY:
    import numpy as np


class ScoreMemory:
    """
    Shared Causal Blackboard for multi-agent synchronization.
    Maintains the collective schema with linear O(n) time complexity
    and constant O(1) memory overhead per token/bar.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        schema_dim: int = 64,
        seed: int = 42
    ):
        self.hidden_dim = hidden_dim
        self.schema_dim = schema_dim
        
        # Cumulative state cache (Causal Mean)
        self.cached_cumsum: Optional[List[float]] = None
        self.cached_counts: float = 0.0
        self.step_count: int = 0
        
        # Projection weights (deterministic pseudo-random initialization)
        rng = random.Random(seed)
        scale_s = 1.0 / math.sqrt(hidden_dim)
        # W_s: [schema_dim, hidden_dim]
        self.W_s = [[rng.uniform(-scale_s, scale_s) for _ in range(hidden_dim)] for _ in range(schema_dim)]
        self.b_s = [0.0 for _ in range(schema_dim)]
        
        # W_gate: [hidden_dim * 2] -> 1 (adaptive gating)
        scale_g = 1.0 / math.sqrt(hidden_dim * 2)
        self.w_gate = [rng.uniform(-scale_g, scale_g) for _ in range(hidden_dim * 2)]
        self.b_gate = -0.5  # Soft bias: starts with balanced attention (~0.38)
        
        # History for auditing
        self.schema_history: List[List[float]] = []

    def reset(self):
        """Resets the collective blackboard state (e.g. at the start of a new problem session)."""
        self.cached_cumsum = None
        self.cached_counts = 0.0
        self.step_count = 0
        self.schema_history.clear()

    def update_cumulative(self, vector: VectorLike, weight: float = 1.0) -> List[float]:
        """
        Incrementally accumulates an agent vector into the causal global context.
        Returns the updated normalized causal context vector c(t) in R^{hidden_dim}.
        """
        vec = to_list(vector)
        if len(vec) != self.hidden_dim:
            # If dimensions differ, adaptively pad or project
            vec = self._adapt_dimension(vec, self.hidden_dim)

        if self.cached_cumsum is None:
            self.cached_cumsum = [x * weight for x in vec]
            self.cached_counts = weight
        else:
            self.cached_cumsum = [c + x * weight for c, x in zip(self.cached_cumsum, vec)]
            self.cached_counts += weight

        self.step_count += 1
        safe_counts = max(1.0, self.cached_counts)
        context = [c / safe_counts for c in self.cached_cumsum]
        return context

    def get_global_schema(self) -> List[float]:
        """
        Projects the current cumulative context into the compact Global Schema s in R^{schema_dim}.
        Uses tanh non-linearity: s = tanh(W_s * c + b_s).
        """
        if self.cached_cumsum is None or self.cached_counts <= 0:
            return [0.0] * self.schema_dim

        safe_counts = max(1.0, self.cached_counts)
        context = [c / safe_counts for c in self.cached_cumsum]
        
        schema = []
        for row, bias in zip(self.W_s, self.b_s):
            act = dot_product(row, context) + bias
            schema.append(math.tanh(act))
            
        self.schema_history.append(schema)
        return schema

    def compute_adaptive_gate(self, agent_vector: VectorLike) -> float:
        """
        Computes the Adaptive Attention Gate g_i in [0.0, 1.0] for an agent.
        g_i represents how much trust the agent should assign to the collective score
        vs maintaining its own internal state:
          g_i = sigmoid(w_gate^T [x_i; c(t)] + b_gate)
        """
        vec = to_list(agent_vector)
        if len(vec) != self.hidden_dim:
            vec = self._adapt_dimension(vec, self.hidden_dim)

        if self.cached_cumsum is None or self.cached_counts <= 0:
            context = [0.0] * self.hidden_dim
        else:
            safe_counts = max(1.0, self.cached_counts)
            context = [c / safe_counts for c in self.cached_cumsum]

        combined = vec + context  # [hidden_dim * 2]
        logit = dot_product(self.w_gate, combined) + self.b_gate
        # Numerically safe sigmoid
        if logit > 20.0:
            return 1.0
        elif logit < -20.0:
            return 0.0
        gate = 1.0 / (1.0 + math.exp(-logit))
        return clamp(gate)

    def reflect_schema(self, agent_vector: VectorLike) -> Tuple[List[float], float]:
        """
        Enriches an agent's state vector under the influence of the global schema:
          x_enhanced = (1 - g_i) * x_i + g_i * reflection(s, x_i)
        Returns:
          (enhanced_vector, gate_value)
        """
        vec = to_list(agent_vector)
        if len(vec) != self.hidden_dim:
            vec = self._adapt_dimension(vec, self.hidden_dim)

        gate = self.compute_adaptive_gate(vec)
        schema = self.get_global_schema()

        # Schema projection back to hidden_dim
        # Broadcast schema dimension across hidden_dim
        refl = []
        for i in range(self.hidden_dim):
            s_val = schema[i % self.schema_dim]
            refl.append(math.tanh(vec[i] * 0.7 + s_val * 0.3))

        # Gated mixing
        enhanced = [(1.0 - gate) * v + gate * r for v, r in zip(vec, refl)]
        return enhanced, gate

    def _adapt_dimension(self, vec: List[float], target_dim: int) -> List[float]:
        """Helper to safely adapt mismatched vector dimensions."""
        if len(vec) >= target_dim:
            return vec[:target_dim]
        # Repeat or zero-pad
        return vec + [0.0] * (target_dim - len(vec))
