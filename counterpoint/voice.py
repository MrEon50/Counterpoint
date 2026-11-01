"""
Voice: Polyphonic Agent Representation in the Counterpoint Mesh.
===============================================================

Wraps any AI agent (Ollama, vLLM, OpenAI, LangChain, or custom Python function)
and connects it to the Counterpoint synchronization bus.
"""

from __future__ import annotations
import math
from typing import Optional, Dict, Any, Callable, Union, List, Tuple

from counterpoint.core.types import (
    AgentState, FrictionVector, VoiceRole, clamp
)


class Voice:
    """
    An agent voice playing in the counterpoint polyphony.
    Exchanges compact states and typed friction with the central mesh.
    """
    def __init__(
        self,
        name: str,
        role: VoiceRole = VoiceRole.GENERAL,
        mesh: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.role = role
        self.mesh = mesh
        self.metadata = metadata or {}
        
        # State tracking
        self.current_state: Optional[AgentState] = None
        self.history: List[AgentState] = []

    def get_context(self) -> Dict[str, Any]:
        """Fetches the global score and personal attention gate from the mesh."""
        if not self.mesh:
            return {"schema_vector": [], "adaptive_gate": 0.5, "latest_resolution": None}
        return self.mesh.get_schema_for(self.name)

    def submit(
        self,
        content: str,
        latent_vector: Optional[List[float]] = None,
        friction: Optional[FrictionVector] = None,
        policy_logits: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """
        Prepares and registers an agent's submission for the current Bar.
        """
        f_vec = friction or FrictionVector()
        state = AgentState(
            voice_id=self.name,
            content=content,
            latent_vector=latent_vector,
            friction=f_vec,
            policy_logits=policy_logits,
            metadata=metadata or {}
        )
        self.current_state = state
        self.history.append(state)
        return state

    def step(
        self,
        task: str,
        handler_fn: Optional[Callable[[str, Dict[str, Any]], Union[str, Tuple[str, FrictionVector]]]] = None
    ) -> AgentState:
        """
        Executes a turn: pulls collective context from mesh, invokes the agent's
        internal logic/handler, and prepares the AgentState for the Bar.
        """
        ctx = self.get_context()
        
        if handler_fn is not None:
            result = handler_fn(task, ctx)
            if isinstance(result, tuple) and len(result) == 2:
                content, f_vec = result
            else:
                content = str(result)
                f_vec = self._estimate_friction_from_text(content)
        else:
            content = f"[{self.name}] processed: {task}"
            f_vec = FrictionVector(epistemic=0.1, axiological=0.0, affective=0.2, surprise=0.1)

        # Generate pseudo-latent embedding from text characters if no vector provided
        pseudo_vec = self._text_to_pseudo_vector(content, dim=getattr(self.mesh, "hidden_dim", 256))
        return self.submit(content=content, latent_vector=pseudo_vec, friction=f_vec)

    def _estimate_friction_from_text(self, text: str) -> FrictionVector:
        """Heuristic fallback to estimate friction signals from textual keywords."""
        lower = text.lower()
        
        # Epistemic: contradiction, error, bug, fail
        e_score = 0.1
        if any(w in lower for w in ["error", "bug", "fail", "incorrect", "contradict", "conflict"]):
            e_score = 0.7
        elif any(w in lower for w in ["maybe", "uncertain", "verify", "check"]):
            e_score = 0.4

        # Axiological: violation, limit, unsafe, forbidden
        v_score = 0.0
        if any(w in lower for w in ["unsafe", "violation", "limit exceeded", "forbidden", "overflow"]):
            v_score = 0.8

        # Affective: intensity
        a_score = clamp(len(text) / 1000.0, min_val=0.1, max_val=0.6)
        
        return FrictionVector(epistemic=e_score, axiological=v_score, affective=a_score, surprise=0.1)

    def _text_to_pseudo_vector(self, text: str, dim: int = 256) -> List[float]:
        """Simple deterministic zero-dependency hash projection for text to vector."""
        vec = [0.0] * dim
        if not text:
            return vec
        for i, char in enumerate(text):
            idx = (ord(char) * 31 + i) % dim
            vec[idx] += 1.0
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            return [x / norm for x in vec]
        return vec
