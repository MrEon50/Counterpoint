# 🎼 Counterpoint (CP) — Multi-Agent Communication & Synchronization Mesh

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-44%2F44%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20(pure%20python%20%2B%20optional%20numpy)-success.svg)]()

> **Lekka, neuro-symboliczna magistrala komunikacji i synchronizacji wieloagentowej (Communication Middleware), która zastępuje powolny, kosztowny „czat tekstowy” między modelami LLM zwięzłą wymianą wektorów tarcia poznawczego i automatyczną syntezą dialektyczną.**

---

## 💡 Czym jest Counterpoint i jaki problem rozwiązuje?

### Problem klasycznych systemów Multi-Agent (LangGraph, CrewAI, AutoGen):
1. **Token Bloat ("Chattering Agents"):** Agenci rozmawiają ze sobą pełnymi akapitami tekstu. Agent A pisze 500 słów, Agent B odpowiada 600 słowami. Po 5 turach system wypala tysiące tokenów na grzecznościowe dyskusje i traci kontekst pierwotnego zadania.
2. **Attention Drift:** W długiej konwersacji uwaga modeli rozmywa się na pobocznych wątkach.
3. **Pętle kłótni lub fałszywy konformizm:** Agenci bez końca spierają się o szczegóły lub od razu kapitulują pod pierwszym argumentem drugiej strony. Brak im mechanizmu **syntezy logicznej**.

### Rozwiązanie Counterpoint:
W muzyce polifonicznej (kontrapunkt) różne głosy prowadzą niezależne melodie, które tworzą harmonijną całość pod rygorem matematycznych interwałów.
W **Counterpoint Mesh**:
- Agenci **nie rozmawiają ze sobą w nieskończoność tekstem naturalnym**.
- Wymieniają zwięzłe wektory stanu i **typowany wektor tarcia poznawczego** $\mathbf{F} = (F_e, F_v, F_a, F_s)$.
- Współdzielą skompresowany schemat globalny $O(n)$ o stałym koszcie pamięciowym $O(1)$ (`ScoreMemory`).
- Gdy moduł wykryje opozycję ($f_{AB} \ge 0.3$), **przerywa dyskusję i automatycznie odpala syntezę dialektyczną $R(A, B) \rightarrow O'$**, weryfikowaną zewnętrznym testem (np. pytest).

---

## 🏛️ Słownik Metafor Kontrapunktu

| Pojęcie muzyczne | Odpowiednik w Counterpoint | Rola w architekturze |
|---|---|---|
| **Voice** | Agent (instancja LLM / worker) | Coder, Critic, Planner, Verifier |
| **Interval** | Wektor Tarcia $\mathbf{F} = (F_e, F_v, F_a, F_s)$ | Typowane napięcie poznawcze w $[0.0, 1.0]$ |
| **Score** | Causal Blackboard $O(n)$ (`ScoreMemory`) | Skompresowany schemat globalny zespołu |
| **Bar** | Takt / Krok synchronizacji | Jednoczesny przebieg stanu wszystkich głosów |
| **Resolution** | Synteza dialektyczna $R(A, B) \rightarrow O'$ | Wyjście z impasu przez stworzenie wyższej abstrakcji |
| **Cadenza** | Zewnętrzna Kotwica Programowa | Test jednostkowy (pytest / parser AST), $v_{\text{ext}} \in \{0, 1\}$ |

---

## 🚀 Szybki Start w 15 Linijkach

Counterpoint jest **całkowicie bez zależności** (pure Python, opcjonalne przyspieszenie NumPy).

```python
from counterpoint import CounterpointMesh, VoiceRole, FrictionVector

# 1. Inicjalizacja magistrali
mesh = CounterpointMesh(hidden_dim=64, schema_dim=16, opposition_threshold=0.30)

# 2. Rejestracja agentów
coder = mesh.register_voice("Coder", role=VoiceRole.GENERATOR)
critic = mesh.register_voice("Critic", role=VoiceRole.CRITIC)

# 3. Zgłoszenie propozycji kodu i uwag krytyka (z wektorami tarcia)
state_coder = coder.submit(
    content="def add(a, b): return a - b", # Bug!
    latent_vector=[1.0] * 64,
    friction=FrictionVector(epistemic=0.2)
)
state_critic = critic.submit(
    content="Krytyczny błąd: funkcja add odejmuje zamiast dodawać!",
    latent_vector=[-1.0] * 64, # Przeciwstawny wektor
    friction=FrictionVector(epistemic=0.9)
)

# 4. Takt synchronizacji z zewnętrznym testem jednostkowym
bar = mesh.tick(
    voice_states={"Coder": state_coder, "Critic": state_critic},
    verifier_fn=lambda code: "return a + b" in code
)

print(f"Tarcie zespołu: {bar.collective_friction:.3f}")
print(f"Zaoszczędzone tokeny: ~{bar.estimated_tokens_saved}")
print(f"Czy rozwiązane?: {mesh.is_resolved()}")
```

---

## 🔌 Przewodnik Integracji z Istniejącymi Architekturami

### Wariant 1: Integracja z LangGraph (Drop-in Node)

Zastąp pętlę wiadomości w LangGraph gotowym węzłem `LangGraphCounterpointNode`:

```python
from langgraph.graph import StateGraph, END
from counterpoint import CounterpointMesh
from counterpoint.adapters import LangGraphCounterpointNode

mesh = CounterpointMesh(opposition_threshold=0.30)

# 1. Definiujemy zewnętrzny test weryfikacyjny (np. pytest)
def my_pytest_anchor(content: str) -> bool:
    return "def solve(" in content and "assert " not in content

# 2. Tworzymy węzeł Counterpoint
cp_node = LangGraphCounterpointNode(
    mesh=mesh,
    verifier_fn=my_pytest_anchor,
    voice_state_keys={"Coder": "coder_output", "Critic": "critic_output"}
)

# 3. Budujemy graf LangGraph
workflow = StateGraph(dict)
workflow.add_node("agent_coder", lambda s: {"coder_output": "def solve(): return 42"})
workflow.add_node("agent_critic", lambda s: {"critic_output": "Sprawdź warunki brzegowe!"})
workflow.add_node("counterpoint_sync", cp_node)

workflow.add_edge("agent_coder", "agent_critic")
workflow.add_edge("agent_critic", "counterpoint_sync")

# Warunkowe wyjście: jeśli CP dokonało syntezy -> END; jeśli nie -> kolejna runda
workflow.add_conditional_edges(
    "counterpoint_sync",
    cp_node.should_continue,
    {"resolved": END, "continue": "agent_coder"}
)
```

---

### Wariant 2: Integracja z lokalnymi modelami Ollama / vLLM / OpenAI

Użyj `StandardAgentAdapter`, aby przekazywać globalny kontekst Counterpoint do dowolnego promptu:

```python
import requests
from counterpoint import CounterpointMesh, Voice
from counterpoint.adapters import StandardAgentAdapter

mesh = CounterpointMesh()
voice = mesh.register_voice("LlamaVoice")

def ollama_generate(prompt: str, context: dict) -> str:
    # Wstrzyknięcie globalnego schematu i wskaźnika uwagi (Adaptive Gate)
    gate = context.get("adaptive_gate", 0.5)
    enhanced_prompt = (
        f"[Globalny Schemat Zespołu (Waga Uwagi: {gate:.2f})]\n"
        f"Zadanie: {prompt}\n"
    )
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3:8b",
        "prompt": enhanced_prompt,
        "stream": False
    })
    return res.json()["response"]

adapter = StandardAgentAdapter(voice, llm_fn=ollama_generate)
state = adapter.run("Napisz funkcję walidacji PESEL w Pythonie")
```

---

## 🔬 Matematyczne Podstawy Modułu

### 1. Causal Cumulative Pooling $O(n)$ (`ScoreMemory`)
Agenci aktualizują wspólny kontekst bez kwadratowej uwagi:
$$c(t) = \frac{\sum_{\tau=0}^t x_\tau \odot m_\tau}{\max(1.0, \sum_{\tau=0}^t m_\tau)}, \quad s(t) = \tanh(W_s \cdot c(t) + b_s)$$
Każdy agent otrzymuje wagę uwagi (Adaptive Gate):
$$g_i(t) = \sigma(w_g^\top [x_i(t); c(t)] + b_g) \in [0.0, 1.0]$$

### 2. Sprzężona Norma Wektora Tarcia $\|F\|_G$ (`FrictionBus`)
Cztery osie: $F_e$ (epistemiczne), $F_v$ (aksjologiczne), $F_a$ (afektywne), $F_s$ (zaskoczenie):
$$\|F\|_G = \sqrt{\mathbf{F}^\top G \mathbf{F}}, \quad G = (1-\kappa_N)I + \kappa_N \mathbf{1}\mathbf{1}^\top$$
$$N = \frac{\|F\|_G}{\sqrt{(1-\kappa_N)n + \kappa_N n^2}}, \quad n=4$$
Kolektywne tarcie uwzględnia dyspersję (zdrową specjalizację agentów):
$$N_{\text{col}} = \frac{1}{n_a} \sum_{i=1}^{n_a} N_i + \gamma_{\text{couple}} \max_{i < j} |N_i - N_j|$$

### 3. Warunek Stabilności Asymptotycznej ODE
Ewolucja tarcia w czasie:
$$F_i(t+1) = \text{clip}_{[0, 1]}\left( F_i(t) + \eta \left[ \sigma_i(t) - \delta F_i(t) + \kappa \left(\sum_{j \neq i} F_j(t)\right)(1 - F_i(t)) \right] \right)$$
Moduł automatycznie wymusza kryterium stabilności: $\delta > 3\kappa$.

### 4. Dialektyczna Synteza $R(A, B) \rightarrow O'$ i Wskaźnik Piękna (`ResolutionEngine`)
Wykrycie opozycji:
$$f_{AB} = \frac{1 - \cos(h_A, h_B)}{2} \ge 0.30$$
Wskaźnik piękna (Beauty Index):
$$B(A, B) = \frac{\frac{1}{2}(\text{depth}_A + \text{depth}_B)}{f_{AB} + 0.05 n_{\text{act}} + 0.10}$$
Warunek akceptacji syntezy:
$$\Delta_{\text{syn}} = (N_{\text{przed}} - N_{\text{po}}) - \mu \cdot \text{complexity} > 0 \quad \wedge \quad v_{\text{ext}} = 1$$

### 5. Równowaga Harmonii $\Phi$ i Ochrona przed Goodhartem (`EquilibriumGovernor`)
$$\Phi = \frac{\sum (\text{Polarity}_i \cdot w_i \cdot (\text{Priority}_i + 1) \cdot \text{Magnitude}_i)}{\sum (w_i \cdot (\text{Priority}_i + 1) \cdot \text{Magnitude}_i)} \in [-1.0, 1.0]$$
Luka Goodharta (zapobiega udawaniu spójności przy braku poprawności):
$$\Gamma = \mathbb{E}[1 - N] - \mathbb{E}[\text{poprawne}]$$
Gdy $\Gamma > 0.35$ przy niskiej zdawalności zadań, moduł automatycznie poddaje kwarantannie zbyt optymistyczne reguły decyzyjne.

---

## 🧠 Kalibracja Epistemiczna i Test Litmus (Anti-Compromise Layer)

### Problem: Fałsz Złotego Środka (*Argumentum ad Temperantiam*)

Gdy jeden agent twierdzi *„Ziemia jest kulista"* (prawda), a drugi *„Ziemia jest płaska”* (fałsz), naiwne systemy oparte na uśrednianiu dochodzą do wniosku: *„Ziemia to półkula"*. **To nie jest kompromis — to nowa halucynacja.**

Counterpoint eliminuje ten problem za pomocą dwóch modułów: `EpistemicLedger` i `LitmusEngine`, które tworzą **podwójną bramkę weryfikacji** na szlaku syntezy dialektycznej.

### 1. `EpistemicLedger` — Księga Wiarygodności

Klasa śledząca dynamiczną wagę epistemologiczną $w_i \in [0, 1]$ każdego głosu:

| Mechanizm | Opis |
|---|---|
| **Ground-Truth Probing** | Mikro-testy z twardą, znaną odpowiedzią kalibrują trafność każdego agenta na starcie i cyklicznie. |
| **Asymmetric Truth Gravity** | Synteza nie uśrednia $(A+B)/2$, lecz stosuje ważoną projekcję $w_A \cdot A + w_B \cdot B$. Jeśli luka autorytetu $\Delta w > 0.40$, słabszy głos jest **eliminowany** (zero szarego kompromisu). |
| **Overconfidence Penalty** | Kara za halucynacyjną pewność siebie: $\text{penalty} = \beta_{oc} \cdot c_i$ gdy odpowiedź jest błędna. |
| **Epistemic Quarantine** | Głos z $w_i \leq 0.15$ traci prawo głosu decyzyjnego i przechodzi w tryb *Listener*. |

```python
from counterpoint import EpistemicLedger, LitmusEngine

# Inicjalizacja (domyślnie aktywna w CounterpointMesh)
ledger = EpistemicLedger(
    initial_authority=0.50,
    accuracy_bonus=0.08,
    overconfidence_beta=0.15,
    dominance_threshold=0.40
)

# Kalibracja: znane pytanie z twardą odpowiedzią
record = ledger.run_calibration_probe(
    voice_id="Coder",
    question_id="math_001",
    expected_answer="42",
    voice_answer="42",       # poprawnie → w_i rośnie o 0.08
    confidence=0.9
)

# Asymetryczna grawitacja: kto dominuje?
w_a, w_b, rejected = ledger.compute_asymmetric_weights("Coder", "Critic")
leader = ledger.get_leader(["Coder", "Critic"])  # → głos z najwyższym w_i
```

### 2. `LitmusEngine` — Przywilej Lidera (Test Sokratyczny)

Głos o **najwyższym autorytecie** ($w_{\max}$) w danym takcie zyskuje prawo wygenerowania jednego testu falsyfikującego:

> *„Twierdzisz, że twoja wersja jest poprawna? Sprawdźmy ją na tym jednym, skrajnym przypadku."*

Workflow w każdym Barze:
1. `EpistemicLedger` identyfikuje **Lidera Epistemologicznego**.
2. Lider rejestruje predykat `probe_fn: (content) -> bool`.
3. Syntezowane $O'$ jest poddawane testowi lidera.
4. Jeśli $O'$ **oblewa** test → synteza odrzucana natychmiast.
5. Jeśli $O'$ **przechodzi** → kontynuacja do standardowej kotwicy zewnętrznej ($v_{\text{ext}}$).

```python
from counterpoint import LitmusEngine, LitmusResult

engine = LitmusEngine()

# Lider (np. "Architect") generuje test falsyfikujący:
engine.register_probe(
    leader_voice_id="Architect",
    probe_fn=lambda code: "return a + b" in code and "assert" not in code,
    description="Funkcja add musi zwracać sumę (edge case: a=0, b=0)",
    probe_name="test_add_edge_case"
)

# Wykonanie testu na propozycji rywala:
result = engine.execute_probe(
    leader_voice_id="Architect",
    target_voice_id="Coder",
    content_to_test="def add(a, b): return a - b"
)
# result.probe_passed → False  (propozycja eliminowana!)
```

### 3. Podwójna Bramka Weryfikacji Syntezy

Warunek akceptacji syntezy $R(A,B) \rightarrow O'$ w `ResolutionEngine`:

$$\text{accepted} = \left( \Delta_{\text{syn}} > 0 \right) \wedge \left( v_{\text{ext}} = 1 \right) \wedge \left( \text{litmus\_passed} \right)$$

| Bramka | Moduł | Rola |
|---|---|---|
| **Gate 1** — Litmus Probe | `LitmusEngine` | Test Sokratyczny lidera (przypadek graniczny / kontrprzykład) |
| **Gate 2** — Programmatic Anchor | `verifier_fn` → $v_{\text{ext}} \in \{0,1\}$ | Twardy test jednostkowy (pytest / parser AST) |

Obie bramki muszą zwrócić `True`, aby synteza została zatwierdzona. Głosy uczestniczące w odrzuconej syntezie tracą wagę epistemologiczną (`synthesis_penalty`), co kaskadowo wzmacnia grawitację prawdy w kolejnych taktach.

---

## 🧪 Uruchomienie Testów i Dema

### 1. Testy Jednostkowe i Integracyjne (100% PASS, 44 testów w 0.15s):
```bash
pytest
# lub:
python -m unittest discover tests
```

### 2. Uruchomienie Dema na Żywo:
```bash
python demo_counterpoint.py
```

Przykładowy wynik z konsoli:
```
[CP] DEMO: Counterpoint Multi-Agent Mesh
[+] Zarejestrowano agentow: ['Coder', 'Critic']

--- BAR 1: Wstepna propozycja kodu i krytyka ---
  [Coder] Wyslal wstepny kod. Tarcie Fe: 0.2
  [Critic] Zglosil sprzecznosc. Tarcie Fe: 0.9
[*] Wyniki Baru 1:
   Kolektywne tarcie (N_col): 0.5544
   Harmonia systemu (Phi)   : -0.3607
   Status syntezy           : ODRZUCONA (Brak poprawki w tescie)

--- BAR 2: Naprawa i udana synteza R(A, B) -> O' ---
[*] Wyniki Baru 2:
   Kolektywne tarcie (N_col): 0.0551 (Spadek z 0.5544)
   Lacznie zaoszczedzono    : ~2241 tokenow wzgledem czatu

[CP] Sukces: Problem rozwiazany w 2 taktach bez zbednej konwersacji tekstowej!
```

---

## 📁 Struktura Modułu

```
counterpoint/
├── __init__.py                 # Eksport: CounterpointMesh, Voice, FrictionVector, etc.
├── mesh.py                     # Centralna magistrala synchronizacji (CounterpointMesh)
├── voice.py                    # Reprezentacja agenta (Voice) z pobieraniem kontekstu
├── core/
│   ├── types.py                # Struktury danych (FrictionVector, AgentState, BarRecord)
│   ├── score_memory.py         # GCL O(n) Causal Blackboard i AdaptiveGate
│   ├── friction_bus.py         # Szyna tarcia poznawczego i dynamika ODE
│   ├── resolution_engine.py    # Bramka syntezy dialektycznej R(A,B)->O' i Beauty Index
│   ├── equilibrium_governor.py # DRM Governor, wskaźnik harmonii Phi i Goodhart Guard
│   ├── epistemic_ledger.py     # Księga Wiarygodności: kalibracja w_i i grawitacja prawdy
│   └── litmus_engine.py        # Przywilej Lidera: test Sokratyczny (falsyfikacja)
└── adapters/
    ├── langgraph_adapter.py    # Gotowy węzeł synchronizujący do LangGraph
    └── standard_agent.py       # Wrapper dla dowolnych funkcji LLM (Ollama, OpenAI)
```

---

## 📄 Licencja

Projekt objęty licencją **MIT License**. Możesz go dowolnie integrować w swoich komercyjnych i badawczych projektach agentowych.
