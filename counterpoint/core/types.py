"""
Data structures, enums, and mathematical helper functions for Counterpoint.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


# ============================================================
# Math Utilities (Zero-Dependency with NumPy Acceleration)
# ============================================================

VectorLike = Union[List[float], "np.ndarray"]

def to_list(vec: VectorLike) -> List[float]:
    """Converts a vector or numpy array to a pure Python float list."""
    if HAS_NUMPY and isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return [float(x) for x in vec]

def dot_product(a: VectorLike, b: VectorLike) -> float:
    """Computes dot product between two vectors."""
    if HAS_NUMPY and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return float(np.dot(a, b))
    la, lb = to_list(a), to_list(b)
    if len(la) != len(lb):
        raise ValueError(f"Vector length mismatch: {len(la)} != {len(lb)}")
    return sum(x * y for x, y in zip(la, lb))

def vector_norm(a: VectorLike) -> float:
    """Computes Euclidean (L2) norm of a vector."""
    if HAS_NUMPY and isinstance(a, np.ndarray):
        return float(np.linalg.norm(a))
    return math.sqrt(sum(x * x for x in to_list(a)))

def cosine_similarity(a: VectorLike, b: VectorLike, eps: float = 1e-9) -> float:
    """Computes cosine similarity in [-1.0, 1.0]."""
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a < eps or norm_b < eps:
        return 0.0
    sim = dot_product(a, b) / (norm_a * norm_b)
    return max(-1.0, min(1.0, float(sim)))

def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a scalar value to [min_val, max_val]."""
    return max(min_val, min(max_val, float(val)))


# ============================================================
# Enums
# ============================================================

class FrictionType(str, Enum):
    """Four orthogonal cognitive friction axes."""
    EPISTEMIC = "epistemic"      # Fe: Inconsistency, logical contradiction, bug in code
    AXIOLOGICAL = "axiological"  # Fv: Safety, value violation, token/memory budget breach
    AFFECTIVE = "affective"      # Fa: Intensity, emotional style, urgency of state
    SURPRISE = "surprise"        # Fs: Total Variation divergence between instinct and reasoning


class VoiceRole(str, Enum):
    """Operational role of an individual agent (Voice) in the polyphonic mesh."""
    GENERATOR = "generator"      # E.g. Primary coder or creative proposer
    CRITIC = "critic"            # E.g. Logic verifier, reviewer, tester
    RETRIEVER = "retriever"      # E.g. Context fetcher, doc finder, memory keeper
    SYNTHESIZER = "synthesizer"  # E.g. Dialectical resolver for deadlocks
    GENERAL = "general"          # Universal agent


# ============================================================
# Friction Vector F = (F_e, F_v, F_a, F_s)
# ============================================================

@dataclass
class FrictionVector:
    """
    Typed Cognitive Friction Vector F = (F_e, F_v, F_a, F_s).
    Every component is strictly normalized in [0.0, 1.0].
    
    Components:
      - epistemic (F_e): Content disagreement or detected logical conflict.
      - axiological (F_v): Constraint or boundary violation penalty.
      - affective (F_a): Agent tension / style modulation signal.
      - surprise (F_s): Divergence TV(pi_f, pi_s) between fast instinct and slow CoT.
    """
    epistemic: float = 0.0
    axiological: float = 0.0
    affective: float = 0.0
    surprise: float = 0.0

    def __post_init__(self):
        self.epistemic = clamp(self.epistemic)
        self.axiological = clamp(self.axiological)
        self.affective = clamp(self.affective)
        self.surprise = clamp(self.surprise)

    def as_list(self) -> List[float]:
        return [self.epistemic, self.axiological, self.affective, self.surprise]

    def as_dict(self) -> Dict[str, float]:
        return {
            "F_e": round(self.epistemic, 4),
            "F_v": round(self.axiological, 4),
            "F_a": round(self.affective, 4),
            "F_s": round(self.surprise, 4),
        }

    def compute_norm(self, kappa_N: float = 0.25) -> float:
        """
        Computes coupled metric norm N in [0.0, 1.0] using coupling matrix G:
          G = (1 - kappa_N) * I + kappa_N * 1 * 1^T
          N = sqrt(F^T * G * F) / sqrt((1 - kappa_N)*n + kappa_N*n^2), where n=4.
        """
        n = 4.0
        f = self.as_list()
        sum_f = sum(f)
        sum_sq = sum(x * x for x in f)
        
        # F^T * G * F = (1 - kappa_N)*sum(F_i^2) + kappa_N*(sum(F_i))^2
        fgf = (1.0 - kappa_N) * sum_sq + kappa_N * (sum_f ** 2)
        denom = math.sqrt((1.0 - kappa_N) * n + kappa_N * (n ** 2))
        
        if denom <= 0.0:
            return 0.0
        return clamp(math.sqrt(max(0.0, fgf)) / denom)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> FrictionVector:
        return cls(
            epistemic=data.get("F_e", data.get("epistemic", 0.0)),
            axiological=data.get("F_v", data.get("axiological", 0.0)),
            affective=data.get("F_a", data.get("affective", 0.0)),
            surprise=data.get("F_s", data.get("surprise", 0.0)),
        )


# ============================================================
# Agent State & Communication Containers
# ============================================================

@dataclass
class AgentState:
    """
    Representation submitted by an agent (Voice) in a Bar (tick).
    Allows agents to exchange compact semantic vectors + key artifacts
    without multi-paragraph conversational bloat.
    """
    voice_id: str
    content: str = ""                         # Text artifact (e.g. code snippet, hypothesis)
    latent_vector: Optional[List[float]] = None # Semantic embedding (dim H, e.g. 256 or 768)
    friction: FrictionVector = field(default_factory=FrictionVector)
    policy_logits: Optional[List[float]] = None # Action distribution pi_f for TV surprise calculation
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SynthesisResult:
    """
    Result of the dialectical resolution operator R(A, B) -> O'.
    """
    triggered: bool
    voice_a: str
    voice_b: str
    opposition_score: float                   # f_AB in [0.0, 1.0]
    beauty_index: float                       # B(A, B)
    synthesized_content: Optional[str] = None # O' generated artifact
    synthesized_vector: Optional[List[float]] = None
    delta_syn: float = 0.0                    # Synthesis gain: N_before - N_after - cost
    external_verdict: int = 0                 # Programmatic anchor result v_ext in {0, 1}
    accepted: bool = False
    message: str = ""


@dataclass
class BarRecord:
    """
    Complete audit trail of a single coordination tick (Bar).
    """
    bar_id: int
    timestamp: float
    voices_active: List[str]
    collective_friction: float                # N_col in [0.0, 1.0]
    system_harmony: float                     # Phi in [-1.0, 1.0]
    per_voice_friction: Dict[str, Dict[str, float]]
    adaptive_gates: Dict[str, float]          # g_i in [0.0, 1.0] per voice
    resolution: Optional[SynthesisResult] = None
    estimated_tokens_saved: int = 0           # Estimated token economy vs standard chat
