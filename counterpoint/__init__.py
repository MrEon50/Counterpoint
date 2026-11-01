"""
counterpoint: Neuro-Symbolic Multi-Agent Communication & Synchronization Middleware
===================================================================================

Counterpoint (CP) replaces verbose, token-burning natural language chattering
between AI agents with a structured, vector-based synchronization mesh.

Key Architectural Metaphors:
- Voice: An individual agent instance (Coder, Critic, Planner, etc.).
- Interval: The typed cognitive friction vector F = (F_e, F_v, F_a, F_s).
- Score: The shared O(n) global context blackboard (GCL).
- Resolution: Dialectical synthesis R(A, B) -> O' resolving agent deadlocks.
- Bar: A synchronous coordination step (tick).
- Cadenza: A GO/NO-GO programmatic verification gate.

Zero-dependency core (pure Python with automatic NumPy acceleration).
"""

from counterpoint.core.types import (
    FrictionVector,
    AgentState,
    SynthesisResult,
    BarRecord,
    VoiceRole,
    FrictionType
)
from counterpoint.core.score_memory import ScoreMemory
from counterpoint.core.friction_bus import FrictionBus
from counterpoint.core.resolution_engine import ResolutionEngine
from counterpoint.core.equilibrium_governor import EquilibriumGovernor
from counterpoint.mesh import CounterpointMesh
from counterpoint.voice import Voice

__version__ = "1.0.0"

__all__ = [
    "CounterpointMesh",
    "Voice",
    "FrictionVector",
    "AgentState",
    "SynthesisResult",
    "BarRecord",
    "VoiceRole",
    "FrictionType",
    "ScoreMemory",
    "FrictionBus",
    "ResolutionEngine",
    "EquilibriumGovernor",
    "__version__",
]
