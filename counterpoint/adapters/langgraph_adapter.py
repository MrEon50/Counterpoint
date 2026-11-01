"""
LangGraphCounterpointNode: Drop-in synchronization node for LangGraph StateGraphs.
================================================================================

Replaces circular node-to-node message ping-pong in LangGraph with a single
Counterpoint synchronization step that breaks loops via dialectical synthesis.
"""

from __future__ import annotations
from typing import Dict, Any, Callable, Optional, List

from counterpoint.mesh import CounterpointMesh
from counterpoint.core.types import AgentState, FrictionVector


class LangGraphCounterpointNode:
    """
    Drop-in node for LangGraph workflows.
    
    Usage:
        mesh = CounterpointMesh()
        sync_node = LangGraphCounterpointNode(mesh, verifier_fn=my_test_runner)
        
        # In LangGraph:
        workflow.add_node("synchronize_voices", sync_node)
        workflow.add_conditional_edges(
            "synchronize_voices",
            sync_node.should_continue,
            {"continue": "agent_coder", "resolved": "end_node"}
        )
    """
    def __init__(
        self,
        mesh: CounterpointMesh,
        verifier_fn: Optional[Callable[[str], bool]] = None,
        voice_state_keys: Optional[Dict[str, str]] = None
    ):
        self.mesh = mesh
        self.verifier_fn = verifier_fn
        # Map of voice_name -> state_dict_key
        self.voice_state_keys = voice_state_keys or {
            "Coder": "coder_output",
            "Critic": "critic_output"
        }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node execution."""
        voice_states: Dict[str, AgentState] = {}

        for voice_name, key in self.voice_state_keys.items():
            content = str(state.get(key, ""))
            if voice_name not in self.mesh.voices:
                self.mesh.register_voice(voice_name)
            
            voice = self.mesh.voices[voice_name]
            f_vec = state.get(f"{key}_friction")
            if not isinstance(f_vec, FrictionVector):
                f_vec = voice._estimate_friction_from_text(content)
                
            p_vec = voice._text_to_pseudo_vector(content, dim=self.mesh.hidden_dim)
            voice_states[voice_name] = AgentState(
                voice_id=voice_name,
                content=content,
                latent_vector=p_vec,
                friction=f_vec
            )

        # Execute Counterpoint Bar
        bar_record = self.mesh.tick(voice_states, verifier_fn=self.verifier_fn)

        # Return updated LangGraph state
        new_state = dict(state)
        new_state["counterpoint_bar"] = bar_record.bar_id
        new_state["counterpoint_collective_friction"] = bar_record.collective_friction
        new_state["counterpoint_harmony"] = bar_record.system_harmony
        new_state["counterpoint_is_resolved"] = self.mesh.is_resolved()

        if bar_record.resolution and bar_record.resolution.accepted:
            new_state["counterpoint_resolution"] = bar_record.resolution.synthesized_content
            new_state["final_output"] = bar_record.resolution.synthesized_content

        return new_state

    def should_continue(self, state: Dict[str, Any]) -> str:
        """Conditional routing function for LangGraph."""
        if state.get("counterpoint_is_resolved", False):
            return "resolved"
        # If bars exceed limit (e.g. 10), break to prevent infinite loops
        if state.get("counterpoint_bar", 0) >= 10:
            return "resolved"
        return "continue"
