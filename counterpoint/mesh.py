"""
CounterpointMesh: The Central Multi-Agent Synchronization Mesh.
==============================================================

Orchestrates multi-agent communication without natural language chat bloat:
1. GCL-based ScoreMemory for O(1) causal context sharing.
2. FrictionBus for 4-axis tension tracking.
3. ResolutionEngine for dialectical synthesis R(A, B) -> O'.
4. EquilibriumGovernor for polarized stability and anti-Goodhart guarding.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Callable, Any, Tuple, Union

from counterpoint.core.types import (
    AgentState, FrictionVector, SynthesisResult, BarRecord, VoiceRole, clamp
)
from counterpoint.core.score_memory import ScoreMemory
from counterpoint.core.friction_bus import FrictionBus
from counterpoint.core.resolution_engine import ResolutionEngine
from counterpoint.core.equilibrium_governor import EquilibriumGovernor
from counterpoint.core.epistemic_ledger import EpistemicLedger
from counterpoint.core.litmus_engine import LitmusEngine


class CounterpointMesh:
    """
    Central Coordinator / Service Mesh for multi-agent synchronization.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        schema_dim: int = 64,
        opposition_threshold: float = 0.30,
        complexity_penalty: float = 0.05,
        target_harmony: float = 0.0,
        harmony_tolerance: float = 0.30,
        synthesizer_fn: Optional[Callable[[str, str, Dict[str, Any]], str]] = None,
        enable_epistemic_calibration: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.schema_dim = schema_dim
        
        # Epistemic Calibration & Litmus Engine (Anti-Compromise)
        self.ledger = EpistemicLedger() if enable_epistemic_calibration else None
        self.litmus_engine = LitmusEngine() if enable_epistemic_calibration else None
        
        # Subsystems
        self.score_memory = ScoreMemory(hidden_dim=hidden_dim, schema_dim=schema_dim)
        self.friction_bus = FrictionBus()
        self.resolution_engine = ResolutionEngine(
            opposition_threshold=opposition_threshold,
            complexity_penalty=complexity_penalty,
            synthesizer_fn=synthesizer_fn,
            ledger=self.ledger,
            litmus_engine=self.litmus_engine
        )
        self.governor = EquilibriumGovernor(
            target_harmony=target_harmony,
            tolerance=harmony_tolerance
        )

        # Registry of voices & execution logs
        self.voices: Dict[str, Any] = {}
        self.bar_history: List[BarRecord] = []
        self._current_bar: int = 0
        self._is_resolved: bool = False
        self._latest_synthesis: Optional[SynthesisResult] = None

    def register_voice(
        self,
        name: str,
        role: Union[VoiceRole, str] = VoiceRole.GENERAL,
        **metadata
    ) -> Any:
        """
        Registers an agent voice into the mesh.
        Returns the registered voice instance.
        """
        from counterpoint.voice import Voice
        if isinstance(role, str):
            try:
                role = VoiceRole(role.lower())
            except ValueError:
                role = VoiceRole.GENERAL

        voice = Voice(name=name, role=role, mesh=self, metadata=metadata)
        self.voices[name] = voice
        
        # Register voice in EpistemicLedger
        if self.ledger is not None:
            self.ledger.register_voice(name)
        
        return voice

    def get_schema_for(self, voice_name: str) -> Dict[str, Any]:
        """
        Retrieves the global score context and personal adaptive gate for a voice.
        """
        schema = self.score_memory.get_global_schema()
        voice_state = self.friction_bus._voice_states.get(voice_name)
        
        gate = 0.5
        if voice_state and voice_state.latent_vector:
            gate = self.score_memory.compute_adaptive_gate(voice_state.latent_vector)

        return {
            "schema_vector": schema,
            "adaptive_gate": round(gate, 4),
            "collective_friction": round(self.friction_bus.compute_collective_tension(), 4),
            "system_harmony": round(self.governor.calculate_system_harmony(), 4),
            "latest_resolution": self._latest_synthesis.synthesized_content if self._latest_synthesis and self._latest_synthesis.accepted else None
        }

    def tick(
        self,
        voice_states: Dict[str, AgentState],
        verifier_fn: Optional[Callable[[str], bool]] = None
    ) -> BarRecord:
        """
        Executes a single coordination Bar (synchronous tick across all active voices).
        
        Args:
          voice_states: Map of voice_name -> AgentState for this bar.
          verifier_fn: Optional Programmatic Anchor function v_ext(content) -> bool.
          
        Returns:
          BarRecord detailing collective tension, harmony, gating, and synthesis.
        """
        self._current_bar += 1
        active_names = list(voice_states.keys())
        gates: Dict[str, float] = {}
        per_voice_friction: Dict[str, Dict[str, float]] = {}

        # Clear litmus probes from previous bar
        if self.litmus_engine is not None:
            self.litmus_engine.clear_probes()

        # 1. Ingest states into FrictionBus & ScoreMemory
        for name, state in voice_states.items():
            self.friction_bus.register_step(name, state)
            per_voice_friction[name] = state.friction.as_dict()

            if state.latent_vector:
                self.score_memory.update_cumulative(state.latent_vector)
                g = self.score_memory.compute_adaptive_gate(state.latent_vector)
            else:
                g = 0.5
            gates[name] = round(g, 4)

        # 2. Compute Collective Tension N_col
        n_col = self.friction_bus.compute_collective_tension()

        # 3. Detect Pairwise Opposition & Attempt Synthesis
        resolution_result: Optional[SynthesisResult] = None
        max_opposition = 0.0
        candidate_pair: Optional[Tuple[str, str]] = None

        if len(active_names) >= 2:
            for i in range(len(active_names)):
                for j in range(i + 1, len(active_names)):
                    name_a, name_b = active_names[i], active_names[j]
                    opp = self.friction_bus.compute_pairwise_opposition(name_a, name_b)
                    if opp > max_opposition:
                        max_opposition = opp
                        candidate_pair = (name_a, name_b)

            if candidate_pair and max_opposition >= self.resolution_engine.opposition_threshold:
                name_a, name_b = candidate_pair
                state_a = voice_states[name_a]
                state_b = voice_states[name_b]

                resolution_result = self.resolution_engine.attempt_resolution(
                    voice_a=name_a,
                    state_a=state_a,
                    voice_b=name_b,
                    state_b=state_b,
                    opposition_score=max_opposition,
                    collective_tension_before=n_col,
                    verifier_fn=verifier_fn,
                    context={"bar_id": self._current_bar, "voice_a": name_a, "voice_b": name_b}
                )

                if resolution_result.accepted:
                    self._is_resolved = True
                    self._latest_synthesis = resolution_result
                    # Inject ratified O' representation into the collective score
                    if resolution_result.synthesized_vector:
                        self.score_memory.update_cumulative(resolution_result.synthesized_vector, weight=2.0)

        # 4. Equilibrium Governor: Harmony & Goodhart Check
        harmony = self.governor.calculate_system_harmony()
        self.governor.balance_weights()
        self.governor.detect_stagnation_and_perturb()
        
        # Record outcome for Goodhart tracking
        success_flag = (resolution_result.accepted if resolution_result else (n_col < 0.35))
        self.governor.record_step_outcome(n_col, success_flag)
        self.governor.check_goodhart_gap()

        # 5. Token Economy Metric: Estimate tokens saved vs verbose chat
        # Baseline chattering agent: ~450 words = ~600 tokens per message per turn
        baseline_tokens = len(active_names) * 600
        # Counterpoint vector exchange: only concise content payload + vectors
        actual_tokens = sum(len(s.content.split()) * 1.3 for s in voice_states.values())
        tokens_saved = max(0, int(baseline_tokens - actual_tokens))

        record = BarRecord(
            bar_id=self._current_bar,
            timestamp=time.time(),
            voices_active=active_names,
            collective_friction=round(n_col, 4),
            system_harmony=round(harmony, 4),
            per_voice_friction=per_voice_friction,
            adaptive_gates=gates,
            resolution=resolution_result,
            estimated_tokens_saved=tokens_saved
        )

        self.bar_history.append(record)
        return record

    def is_resolved(self) -> bool:
        """Returns True if a dialectical resolution O' has been accepted."""
        return self._is_resolved

    def get_latest_resolution(self) -> Optional[SynthesisResult]:
        """Returns the most recent dialectical synthesis outcome."""
        return self._latest_synthesis

    def get_status(self) -> Dict[str, Any]:
        """Summary metrics of the Counterpoint Mesh."""
        total_tokens_saved = sum(b.estimated_tokens_saved for b in self.bar_history)
        n_col = self.friction_bus.compute_collective_tension()
        harmony = self.governor.calculate_system_harmony()
        gamma, hacking = self.governor.check_goodhart_gap()
        
        return {
            "current_bar": self._current_bar,
            "registered_voices": list(self.voices.keys()),
            "collective_tension": round(n_col, 4),
            "system_harmony": round(harmony, 4),
            "goodhart_gap_gamma": round(gamma, 4),
            "is_resolved": self._is_resolved,
            "total_bars_executed": len(self.bar_history),
            "estimated_tokens_saved": total_tokens_saved,
        }
