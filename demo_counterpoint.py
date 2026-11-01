"""
demo_counterpoint.py: Live Demonstration of Counterpoint Multi-Agent Mesh.
===========================================================================

Demonstrates how Counterpoint replaces conversational multi-agent ping-pong
with structured, token-efficient vector synchronization and dialectical synthesis.

Scenario:
  Task: "Zaimplementuj algorytm szybkiego potegowania modularnego w Pythonie."
  - Bar 1: Coder zglasza kod z bledem. Critic zglasza sprzecznosc.
           Magistrala wykrywa tarcie (f_AB >= 0.30). Test jednostkowy nie przechodzi.
           Synteza odrzucona (bezpiecznik v_ext = 0).
  - Bar 2: Coder/Syntetyk nanosi poprawke. Kotwica programowa potwierdza sukces.
           Synteza zaakceptowana (v_ext = 1, Delta_syn > 0). Tarcie spada, problem rozwiazany!
"""

import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from counterpoint import CounterpointMesh, VoiceRole, FrictionVector

def run_demo():
    print("=" * 75)
    print("[CP] DEMO: Counterpoint Multi-Agent Mesh")
    print("     Subskrypcja stanu, wektorowe tarcie i dialektyczna synteza R(A,B) -> O'")
    print("=" * 75)

    # 1. Inicjalizacja Magistrali CP
    mesh = CounterpointMesh(
        hidden_dim=64,
        schema_dim=16,
        opposition_threshold=0.30,
        harmony_tolerance=0.30
    )

    # 2. Rejestracja Glosow (Voices)
    coder = mesh.register_voice("Coder", role=VoiceRole.GENERATOR)
    critic = mesh.register_voice("Critic", role=VoiceRole.CRITIC)
    print(f"\n[+] Zarejestrowano agentow: {[v for v in mesh.voices]}")

    # =========================================================================
    # BAR 1 (Takt 1): Wstepna propozycja z bledem
    # =========================================================================
    print("\n" + "=" * 50)
    print("--- BAR 1: Wstepna propozycja kodu i krytyka ---")
    print("=" * 50)
    
    buggy_code = (
        "def mod_exp(base, exp, mod):\n"
        "    res = 1\n"
        "    while exp > 0:\n"
        "        if exp % 2 == 1: res = (res * base) % mod\n"
        "        base = (base * base) % mod\n"
        "        exp //= 2\n"
        "        return res # BUG: przedwczesny return!"
    )
    
    state_coder_1 = coder.submit(
        content=buggy_code,
        latent_vector=[0.8] * 64,
        friction=FrictionVector(epistemic=0.2, axiological=0.0, affective=0.3, surprise=0.1)
    )
    print("  [Coder] Wyslal wstepny kod. Tarcie Fe:", state_coder_1.friction.epistemic)

    critic_feedback_1 = (
        "Blad krytyczny: przedwczesny return w petli while oraz brak obslugi ujemnych wykladnikow."
    )
    state_critic_1 = critic.submit(
        content=critic_feedback_1,
        latent_vector=[-0.8] * 64, # Przeciwstawny wektor
        friction=FrictionVector(epistemic=0.9, axiological=0.7, affective=0.6, surprise=0.2)
    )
    print("  [Critic] Zglosil sprzecznosc. Tarcie Fe:", state_critic_1.friction.epistemic)

    # Zewnetrzny weryfikator (np. pytest)
    def test_runner(code_text: str) -> bool:
        # Wymaga usuniecia 'BUG:' i poprawnej petli
        if "BUG:" in code_text:
            return False
        return "while exp > 0:" in code_text and "res = 1" in code_text and "return res" in code_text

    # Wykonanie Baru 1
    bar_1 = mesh.tick(
        voice_states={"Coder": state_coder_1, "Critic": state_critic_1},
        verifier_fn=test_runner
    )

    print(f"\n[*] Wyniki Baru 1:")
    print(f"   Kolektywne tarcie (N_col): {bar_1.collective_friction:.4f}")
    print(f"   Harmonia systemu (Phi)   : {bar_1.system_harmony:+.4f}")
    print(f"   Bramki uwagi (g_i)       : {bar_1.adaptive_gates}")
    print(f"   Status syntezy           : {'ZAAKCEPTOWANA' if bar_1.resolution and bar_1.resolution.accepted else 'ODRZUCONA (Brak poprawki w tescie)'}")

    # =========================================================================
    # BAR 2 (Takt 2): Uwzglednienie krytyki i poprawna synteza
    # =========================================================================
    print("\n" + "=" * 50)
    print("--- BAR 2: Naprawa i udana synteza R(A, B) -> O' ---")
    print("=" * 50)

    # Coder odbiera globalny schemat i koryguje kod
    fixed_code = (
        "def mod_exp(base, exp, mod):\n"
        "    if mod <= 1: raise ValueError('mod must be > 1')\n"
        "    if exp < 0: raise ValueError('negative exp not supported')\n"
        "    res = 1\n"
        "    base = base % mod\n"
        "    while exp > 0:\n"
        "        if exp % 2 == 1: res = (res * base) % mod\n"
        "        base = (base * base) % mod\n"
        "        exp //= 2\n"
        "    return res"
    )

    state_coder_2 = coder.submit(
        content=fixed_code,
        latent_vector=[0.1] * 64,
        friction=FrictionVector(epistemic=0.05, axiological=0.0, affective=0.1, surprise=0.05)
    )
    print("  [Coder] Wyslal poprawiony kod po analizie schematu globalnego.")

    state_critic_2 = critic.submit(
        content="Walidacja zaliczona: brak petli nieskonczonych, poprawne modulo i asercje brzegowe.",
        latent_vector=[0.12] * 64, # Wektory sa teraz zgodne!
        friction=FrictionVector(epistemic=0.05, axiological=0.0, affective=0.1, surprise=0.05)
    )
    print("  [Critic] Zgodnosc logiczna potwierdzona.")

    # Wykonanie Baru 2
    bar_2 = mesh.tick(
        voice_states={"Coder": state_coder_2, "Critic": state_critic_2},
        verifier_fn=test_runner
    )

    print(f"\n[*] Wyniki Baru 2:")
    print(f"   Kolektywne tarcie (N_col): {bar_2.collective_friction:.4f} (Spadek z {bar_1.collective_friction:.4f})")
    print(f"   Harmonia systemu (Phi)   : {bar_2.system_harmony:+.4f}")
    print(f"   Bramki uwagi (g_i)       : {bar_2.adaptive_gates}")
    print(f"   Lacznie zaoszczedzono    : ~{mesh.get_status()['estimated_tokens_saved']} tokenow wzgledem czatu")

    print("\n" + "=" * 75)
    print(f"[CP] Sukces: Problem rozwiazany w 2 taktach bez zbednej konwersacji tekstowej!")
    print("=" * 75)

if __name__ == "__main__":
    run_demo()
