"""
StandardAgentAdapter: Generic callable / LLM adapter for Counterpoint.
======================================================================

Easily connects any prompt generator, Ollama function, or OpenAI/Anthropic call
to a Counterpoint Voice.
"""

from __future__ import annotations
from typing import Callable, Any, Dict, Optional, Tuple

from counterpoint.core.types import FrictionVector
from counterpoint.voice import Voice


class StandardAgentAdapter:
    """
    Adapts an existing agent function to the Counterpoint Mesh.
    
    Usage:
        def my_llm_coder(prompt: str, context: dict) -> str:
            # e.g. call Ollama or OpenAI
            return "def solution(): return 42"
            
        adapter = StandardAgentAdapter(voice, llm_fn=my_llm_coder)
        state = adapter.run("Implement quicksort")
    """
    def __init__(
        self,
        voice: Voice,
        llm_fn: Callable[[str, Dict[str, Any]], str],
        friction_evaluator_fn: Optional[Callable[[str, str], FrictionVector]] = None
    ):
        self.voice = voice
        self.llm_fn = llm_fn
        self.friction_evaluator_fn = friction_evaluator_fn

    def run(self, task: str) -> Any:
        """Runs the agent with Counterpoint global schema context injected."""
        ctx = self.voice.get_context()
        raw_output = self.llm_fn(task, ctx)
        
        if self.friction_evaluator_fn:
            f_vec = self.friction_evaluator_fn(task, raw_output)
        else:
            f_vec = self.voice._estimate_friction_from_text(raw_output)

        pseudo_vec = self.voice._text_to_pseudo_vector(raw_output, dim=getattr(self.voice.mesh, "hidden_dim", 256))
        return self.voice.submit(content=raw_output, latent_vector=pseudo_vec, friction=f_vec)
