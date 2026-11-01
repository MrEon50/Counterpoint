# Counterpoint (CP) — Analiza architektoniczna i plan ulepszeń

_Data: 11.11.2026 · Zakres: pełny przegląd kodu (`counterpoint/`, `tests/`), weryfikacja testami_

---

## 1. Przeznaczenie projektu

**Counterpoint (CP) to framework koordynacji wielu agentów LLM oparty na metaforze
polifonicznego kontrapunktu muzycznego.** Nie jest aplikacją muzyczną — nazwa i terminologia
(friction, harmony, bar, score, litmus) to spójny model metaforyczny nad mechanizmem
rozstrzygania konfliktów między agentami.

Kluczowe idee:

| Pojęcie z domeny | Znaczenie operacyjne |
|---|---|
| **Voice** (głos) | jeden agent LLM z własnym stanem, autorytetem i wektorem latentnym |
| **Friction vector F = (F_e, F_v, F_a, F_s)** | typowany wektor "tarcia kognitywnego": epistemiczne, aksiologiczne, afektywne, zaskoczenie |
| **Collective tension N_col** | zbiorcze napięcie zespołu agentów (średnia + dyspersja) |
| **Opposition f_AB = (1−cos(h_A,h_B))/2** | semantyczny dystans opozycyjny między wektorami latentnymi |
| **Synthesis / Resolution** | procedura rozstrzygania konfliktu: kandydat O' podwójnie weryfikowany |
| **Asymmetric Truth Gravity (ATG)** | przy dużej różnicy autorytetu — "wygrany bierze wszystko", bez szarego uśredniania |
| **Litmus Probe** | test falsyfikacyjny autorstwa lidera, egzekwowany na kandydacie O' |
| **Programmatic Anchor (v_ext ∈ {0,1})** | twardy gate: zewnętrzna weryfikacja programatyczna (np. test kodu) — GO/NO-GO |
| **Equilibrium Governor / Goodhart guard** | ochrona przed reward-hackingiem: agenci nie mogą "obniżać napięcie" bez realnego efektu |

Model celowo odrzuca uśrednianie kompromisowe ("no grey averaging") na rzecz binarnych,
weryfikowalnych rozstrzyęć — to jego główna cecha wyróżniająca vs. typowe frameworki
multi-agent (debaty → konsensus).

---

## 2. Przegląd konstrukcji

### 2.1 Mapa modułów i przepływ danych

```
Voice.submit(content, vector, friction)
        │
        ▼
Mesh.tick(voice_states, verifier_fn)            [mesh.py]
  ├─ FrictionBus: N_col, f_AB per para          [core/friction_bus.py]
  ├─ ResolutionEngine.attempt_resolution       [core/resolution_engine.py]
  │    ├─ Gate 1: EpistemicLedger (ATG)        [core/epistemic_ledger.py]
  │    ├─ Synthesis O' (synthesizer_fn)
  │    ├─ Gate 2: LitmusEngine.execute_probe   [core/litmus_engine.py]
  │    └─ Gate 3: verifier_fn → v_ext ∈ {0,1}
  ├─ EquilibriumGovernor                       [core/equilibrium_governor.py]
  │    ├─ Phi (harmony), balance_weights
  │    ├─ stagnation check (variance φ)
  │    └─ Goodhart guard: Γ = E[1−N] − E[succ]
  └─ BarRecord → mesh.get_status() / get_schema_for()
```

### 2.2 Mocne strony konstrukcji

1. **Spójny, w pełni zrealizowany model** — każde pojęcie z README ma implementację;
   brak "martwych" komponentów. Zależności czyste (core nie importuje mesh).
2. **Podwójna weryfikacja (litmus + anchor)** to rzadki, wartościowy wzorzec:
   agent nie może sam sobie przyznać racji — wymagany jest zewnętrzny dowód programatyczny.
3. **Goodhart guard z matematycznym uzasadnieniem** (Γ = E[1−N] − E[succ]) — realna
   ochrona przed reward-hackingiem, rzadkość w frameworkach tego typu.
4. **ATG z ledgerem autorytetu i kwarantanną** — kalibracja na oknach historycznych,
   penalizacja overconfidence (β), izolacja niekalibrowanych głosów.
5. **Solidna baza testowa jednostkowa** (12 plików, ~97 asercji) — pokrycie każdego modułu core.
6. **Czysto funkcjonalne rdzenie** (FrictionBus, Governor, Ledger bezstanowe względem I/O),
   co ułatwia testowanie i podmianę backendów LLM.

### 2.3 Słabe strony konstrukcji (ogólne)

- Brak integracji z rzeczywistym LLM (wszystko jest interfejsem pod `content`/`vector`) —
  świadomy wybór, ale semantyka wektorów zależy w 100% od dostarczyciela.
- Brak CI/lint/configu w repo; testy uruchamiane ad hoc.
- Dostęp do stanu prywatnego (`_success_history`, `_phi_history`) z poziomu mesh —
   akceptowalne w tym etapie, ale zaprasza do testów bazujących na implementacji.


---

## 3. Zidentyfikowane defekty — P0 (potwierdzone testami RED)

Testy: **`tests/test_regression_p0.py`** (8 testów; dziś: 6 FAIL = defekty, 2 ok = guardy).
Każdy defekt ma powiązaną poprawkę w sekcji 4.1.

### P0-1 · `resolution_engine.py` (~linia 115–148) — ATG omija Gate 3 ⚠️ najpoważniejszy
W gałęzi *winner-takes-all* wynik jest tworzony z **zadany** `external_verdict=1, accepted=True`,
bez wywołania `verifier_fn`. Skutek: silna teza dominującego agenta jest akceptowana
nawet gdy programatyczny anchor ją obala — dokładnie to, przed czym model ma chronić.
Ponadto ledger nagradza "wygranego" niezależnie od werdyktu.

### P0-2 · `litmus_engine.py` (~linia 88–143) — metadane probu w wspólnych slotach `_last_*`
`register_probe()` zapisuje opis/nazwę lidera X do współdzielonych atrybutów instancji.
`execute_probe()` odczytuje je **niezależnie od tego, czyj prob się wykonuje** → przy ≥2
lidarach z aktywnymi probami ślad audytowy raportuje metadane ostatniego zarejestrowanego
lidera (wyciek między liderami w `LitmusResult.probe_description/probe_fn_name`).

### P0-3 · `equilibrium_governor.py` + `mesh.py` — Phi zapisywane przy każdym wywołaniu
`calculate_system_harmony()` appenduje do `_phi_history`. Jeden Bar wywołuje ją 2×
(`tick` linia 193 + `balance_weights` linia 119) → seria czasowa φ ma 2 próbki na Bar.
Skutek: wariancja stagnacji (okno 5) i okna Goodharta są systematycznie zniekształcone;
długości okien w testach Goodharta opierają się na tym założeniu.

### P0-4 · `mesh.py` linia 198 — flaga sukcesu bez zakotwiczenia
```python
success_flag = (resolution_result.accepted if resolution_result else (n_col < 0.35))
```
Gdy brak rozstrzygnięcia, "sukces" = niskie **zadeklarowane** przez agentów napięcie.
To otwarta dziura Goodharta: agenci uczą się raportować niskie F_e bez wykonania zadania,
a Γ nie wykryje hacku (E[succ] pozostaje wysokie).

---

## 4. Plan ulepszeń

### 4.1 Faza 1 — naprawa P0 (minimalna, zachowawcza)

| # | Plik | Zmiana |
|---|---|---|
| F1 | `resolution_engine.py` | Gałąź ATG: wywołaj `verifier_fn(winner.content)` → `v_ext`; `accepted = (v_ext == 1)`; nagroda/kara w ledgerze **tylko** gdy `v_ext == 1`. Brak verifiera ⇒ `v_ext = 0` (niezakotwiczona teza = NO-GO). |
| F2 | `litmus_engine.py` | Zamiast `_last_*`: słownik `{leader: (probe_fn, description, probe_name)}`; `execute_probe` odczytuje wpis własnego lidera. API bez zmian. |
| F3 | `equilibrium_governor.py`, `mesh.py` | Nowe `get_current_harmony()` (czyste obliczenie, bez appendu); `calculate_system_harmony` = alias czysty; nowy `record_bar_snapshot()` — jedyne miejsce appendu. `balance_weights` używa `get_current_harmony()`. `mesh.tick` wywołuje snapshot 1×/Bar. |
| F4 | `mesh.py` linia 198 | `success_flag = bool(resolution_result and resolution_result.accepted)` — sukces wymaga zakotwiczonego, akceptowanego rozstrzygnięcia. |

Kryterium zakończenia: `python -m unittest tests.test_regression_p0 -v` → **8/8 GREEN**
oraz pełna istniejąca suita (`tests/`) bez regresji.

### 4.2 Faza 2 — spójność semantyczna i hardening (P1)

1. **Litmus w gałęzi ATG** — obecnie Gate 2 działa tylko na ścieżce kompromisowej;
   dla pełnej symetrii "double gate" uruchamiać prob lidera też na wygranym O'.
2. **`SynthesisResult.litmus_verdict` w ATG** — dziś `None`; po pkt 1 realna wartość.
3. **Stabilizacja Goodharta** — Γ liczone jest od pierwszej pary okien; dodać próg
   minimalnej długości historii (np. ≥ 2×window) przed emisją alertu, inaczej pojedynczy
   hałaśliwy Bar potrafi wywołać `goodhart_gap_detected`.
4. **Wektor syntezowany w ATG** — dziś to wektor zwycięzcy; rozważyć blend z wagą
   `w_winner` dla płynności score memory (działa, ale skacze).
5. **CI + lint** — `.github/workflows/ci.yml` (unittest na Python 3.10+), config `ruff`,
   zamrożenie wymagań (`requirements.txt`). Repo nie ma dziś żadnej konfiguracji.

### 4.3 Faza 3 — rozwój produktu (P2)

1. **Adapter LLM** — interfejs `VoiceProvider` z implementacją pod konkretny model
   (embeddingi → latent vectors, odpowiedź → content); dzisiaj to największa luka
   między "frameworkiem" a działającym systemem end-to-end.
2. **Trzy i więcej głosów w rozstrzygnięciu** — `attempt_resolution` jest parowy;
   dla N głosów potrzebna hierarchia (turniej/drzewo) z zachowaniem ATG per para.
3. **Persystencja ledgera/score memory** — stan ginie po restarcie; serializacja JSON/SQLite
   przyda się w dłuższych sesjach wielobarowych.
4. **Dashboard harmonii** — `get_status()` daje komplet danych (φ, N_col, Γ, gates);
   prosta strona HTML/terminal renderująca "partyturę" bar po barze.
5. **Metryka token economy** — dziś szacowana linią 206 mesh.py; realny pomiar z licznika API.

---

## 5. Podsumowanie oceny

| Obszar | Ocena |
|---|---|
| Spójność modelu konceptualnego | **wysoka** — rzadko spotykana kompletność |
| Jakość kodu core | dobra (czytelne, typowane, testowane) |
| Poprawność gate'ów rozstrzygania | **niska** po P0-1/P0-2 — wrażliwe ścieżki omijają weryfikację |
| Ochrona przed Goodhartem | częściowa (dziura P0-4) |
| Gotowość do integracji z LLM | niska (brak adaptera) |

**Rekomendacja:** najpierw Faza 1 (4 małe poprawki, każda < 20 linii, pokryte testami),
potem Faza 2 pkt 5 (CI — tani ubezpieczony przyszłych zmian), a dopiero potem Faza 3.

