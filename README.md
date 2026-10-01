# Eco-Travel Advisor

A conversational sustainable-tourism planner built on **Rasa 3.6** and
**Streamlit**. The bot elicits trip details through an adaptive multi-turn
form, ranks accommodation by proxy sustainability signals from
OpenStreetMap, computes per-transport-mode carbon footprints, and escalates
to a human advisor with full conversation context when it cannot help.

Built as the assessment for the MSc AI module *Advanced Conversational UI
Design & Chatbot Development* (BSBI Berlin / University for the Creative
Arts, cohort G1).

---

## Quick start (local development)

Prerequisites:
- macOS or Linux (Windows works but examples below use bash / zsh syntax)
- Python 3.10
- `pip`
- Optional: Docker Desktop 24+

```bash
# 1. Create and activate the virtual environment
python3.10 -m venv rasa-env
source rasa-env/bin/activate

# 2. Install everything (three requirement files split by service concern)
pip install -r requirements-rasa.txt
pip install -r requirements-actions.txt
pip install -r requirements-streamlit.txt

# 3. Train the model
rasa train

# 4. Pre-fetch OpenStreetMap POI data (optional, one-off)
python scripts/fetch_pois.py

# 5. Start the three processes — each in a separate terminal
rasa run --enable-api --cors "*" --endpoints endpoints.yml   # 5005
rasa run actions                                             # 5055
streamlit run streamlit_app.py                               # 8501

# 6. Open http://localhost:8501
```

---

## Quick start (Docker)

```bash
cp .env.example .env      # optional; only needed for Climatiq
docker compose up --build
# → http://localhost:8501
```

Three services come up together on an internal bridge network:

| Service     | Port | What it does                                       |
|-------------|-----:|----------------------------------------------------|
| `streamlit` | 8501 | Chat UI, renders bot-sent buttons, colour cards    |
| `rasa`      | 5005 | NLU + Core + REST webhook (`/webhooks/rest/webhook`) |
| `actions`   | 5055 | Custom Python actions, external API calls          |

Retrain the model on the host before rebuilding if you change any `*.yml`.

---

## Architecture

```
                  ┌─────────────────────────────┐
                  │  Streamlit chat UI (8501)   │
                  │  colour cards, button       │
                  │  render, handover banner    │
                  └──────────────┬──────────────┘
                                 │ HTTP POST /webhooks/rest/webhook
                                 ▼
    ┌───────────────────────────────────────────────────────────┐
    │  Rasa Core + NLU (5005)                                   │
    │  - DIETClassifier (WhitespaceTokenizer + Regex /          │
    │    LexicalSyntactic / CountVectors featurisers)           │
    │  - FallbackClassifier (threshold 0.4)                     │
    │  - RulePolicy + TEDPolicy + UnexpecTEDIntentPolicy        │
    │  - Two-stage fallback → action_default_fallback           │
    └──────────────┬────────────────────────────────────────────┘
                   │ POST /webhook  (custom action decisions)
                   ▼
    ┌───────────────────────────────────────────────────────────┐
    │  Actions server (5055)                                    │
    │  - ValidateTripPlanningForm (adaptive required_slots +    │
    │    per-slot validators)                                   │
    │  - ActionSuggestEcoHotels (OSM proxy signals, no false    │
    │    certification claims)                                  │
    │  - ActionDescribePlace (Wikipedia REST)                   │
    │  - ActionHumanHandover (full transcript + intent bundle)  │
    │  - ...9 more                                              │
    └──────────────┬────────────────────────────────────────────┘
                   │
    ┌──────────────┴────────┬────────────────┬─────────────────┐
    │ Nominatim             │ Open-Meteo     │ Frankfurter     │
    │ (geocoding, live)     │ (weather)      │ (currency)      │
    │                       │                │                 │
    │ Wikipedia REST        │ Overpass       │ Climatiq        │
    │ (place summary)       │ (POI, OFFLINE  │ (carbon, live;  │
    │                       │  pre-fetched   │  local BEIS     │
    │                       │  to JSON)      │  table if down) │
    └───────────────────────┴────────────────┴─────────────────┘
```

Rasa's own file processing order at training time: `config.yml` → `domain.yml`
→ `data/nlu.yml` → `data/rules.yml` → `data/stories.yml`. At runtime the
Streamlit UI POSTs JSON to the Rasa REST endpoint; Rasa's policies pick the
next action; custom actions are dispatched to the actions server over the
`action_endpoint` URL defined in `endpoints.yml` (or `endpoints.docker.yml`
for compose).

---

## External APIs used

Following the module's revised API guidance (see `Checks/README.md` — Amadeus
for Developers was decommissioned on 17 July 2026):

| API              | Key required | Rate limit               | Called from                      |
|------------------|:------------:|--------------------------|----------------------------------|
| **Nominatim**    | No           | 1 req/sec, User-Agent    | Live from actions server         |
| **Overpass**     | No           | Best-effort              | **Offline** — see below          |
| **Wikipedia**    | No           | Generous                 | Live from actions server         |
| **Open-Meteo**   | No           | Generous                 | Live from actions server         |
| **Frankfurter**  | No           | Generous                 | Live from actions server         |
| **Climatiq**     | Yes (free)   | 2,500 calls/month (free) | Live per-mode carbon, with fallback |

Nominatim and Overpass are volunteer-funded and require an identifying
User-Agent. This project sets `EcoTravelAdvisor/1.0 (barislaca94@gmail.com)`;
change it in `actions/actions.py` (`USER_AGENT`) and
`scripts/fetch_pois.py` if you fork the repo.

### Why Overpass is pre-fetched, not called live

The assignment brief requires responses "under three seconds for critical
interactions." Overpass frequently returns `504 Gateway Timeout` or blocks
under load. Running Overpass live from a conversation turn would violate
that latency budget. Instead, `scripts/fetch_pois.py` queries Overpass once
per supported city and writes the results to
`data/eco_data/<city>_(hotels|transit|attractions).json`. The actions
server reads those JSON files at request time.

Re-run the script when the supported city list changes:

```bash
python scripts/fetch_pois.py                       # everything
python scripts/fetch_pois.py Copenhagen Oslo Paris # subset
```

### Carbon figures: Climatiq, with a local fallback

`estimate_carbon()` in `actions/actions.py` batches one Climatiq request per
answer (2.5 s timeout, results cached per mode and rounded distance). If
`CLIMATIQ_API_KEY` is unset, or the call fails, the local `EMISSION_FACTORS`
table answers instead. Either way the reply names the source it used, so a
local average is never presented as a live lookup.

```bash
cp .env.example .env         # then paste your key into CLIMATIQ_API_KEY
python scripts/climatiq_check.py
```

`climatiq_check.py` estimates 100 passenger-km for every mode and prints the
Climatiq figure next to the local one. Activity IDs are version-specific, so
when Climatiq retires one the script searches for replacements and prints the
candidates. Verified against data version `^21` on 2026-09-16:

| Mode | Climatiq (per 100 pkm) | Local table | Source |
|---|---:|---:|---|
| Train | 3.09 kg | 3.10 kg | BEIS |
| Bus (coach factor) | 3.95 kg | 4.00 kg | BEIS |
| Car | 16.42 kg | 16.40 kg | UBA |
| Flight, short haul | 12.58 kg | 12.60 kg | BEIS |
| Flight, long haul | 11.70 kg | 12.60 kg | BEIS |

Two deliberate choices sit behind that table. `bus` maps to Climatiq's
*coach* factor, because the local-bus factor models urban stop-start driving
rather than the city-to-city journeys this bot compares. Flights switch
activity ID at 3,700 km: BEIS publishes separate short- and long-haul
factors, and the undifferentiated average understates a short hop by about a
third.

---

## Sustainability & ethics

The assignment brief flags **greenwashing** as an ethical risk in tourism
tech. This bot addresses that risk explicitly:

- Hotel data comes from OpenStreetMap, which does **not** carry reliable
  eco-certification tags. The bot **never** claims a specific property is
  certified.
- Instead it computes and displays *proxy* signals: distance to the nearest
  **rail, metro or tram** stop, room count (small = smaller footprint per
  stay), and presence/absence of a `parking` tag.
- Rail distance is graded (under 300 m / 800 m / 1500 m) rather than a single
  threshold. An earlier version counted any `public_transport=stop_position`
  node, which meant a bus stop 20 m away — something nearly every city-centre
  hotel has — and every hotel scored the same. Measuring tag coverage across
  the cached cities showed `parking` present on under 2% of hotels, so it
  almost never contributes; `stars` (37%) is used only as a price proxy, and
  `wheelchair` (42%) is reported for accessibility but never scored.
- Missing tags never lower a score. OSM omissions are far more common than
  OSM falsehoods, so absence is treated as unknown, not as a negative.
- Every hotel response ends with an explicit disclaimer stating the
  signals are proxies and pointing users to independent certification
  sources (Green Key, EU Ecolabel, LEED).
- Carbon offset suggestions link to verified programmes (Gold Standard,
  Atmosfair, myclimate, Klima) and remind the user that offsets should
  *complement* low-emission choices, not replace them.

Data privacy: the bot keeps conversation state in Rasa's in-memory
tracker store for a single session. No persistent user data is written to
disk beyond the on-host trained model. The handover package printed to the
actions log is intended to be forwarded to a Slack/Zendesk-style ticketing
system in production; the demo prints it to stdout so it can be reviewed
without external services.

---

## Testing

### NLU cross-validation

```bash
rasa test nlu --cross-validation --folds 3
```

Artefacts land under `results/`:

- `intent_report.json` — per-intent precision / recall / F1
- `intent_confusion_matrix.png`
- `intent_histogram.png` — confidence distribution
- `DIETClassifier_*` — entity extraction metrics

Current figures (model `20260917-115128-vintage-clone`): **0.721 mean
accuracy** over three 3-fold runs. Repeating the same command on the same data
varies by up to 0.049, so single-run comparisons are meaningless at this
corpus size. The figure is lower than earlier phases (0.772) because two
deliberately broad intents were added — `off_topic` and `bot_challenge` — at
the same time as user-facing behaviour improved; `evaluation/README.md`
explains that trade-off and records every run.

### Dialogue smoke test

```bash
rasa run actions &                                  # the script uses the real action server
python scripts/dialogue_smoke.py models/<model>.tar.gz
```

Drives whole conversations through `Agent.handle_text` — the code path the
server uses — and fails a turn that is slow, silent, or missing its expected
reply. A turn over 30 seconds dumps every thread's Python stack and aborts.

It exists because of a bug no other test could see. `core_fallback_action_name`
was set to `action_two_stage_fallback`, a loop meant for messages the NLU did
not understand. When a message *was* understood but no rule covered it — "What
is 2+2" is classified as `inform` — that loop activated, found nothing to
clarify, closed without replying, and was chosen again: 255 times in 25
seconds, until the chat UI gave up. NLU tests only classify, `rasa test core`
replays stories without the live prediction loop, and the unit tests stub
Rasa out entirely. The fix is `action_core_fallback`, which answers once and
rewinds the message.

### NLU regression set

Cross-validation can only test phrasings that are already in the training
data, which is exactly why it missed the regression that prompted this suite:
adding training examples to three weak intents made `"How are you"` classify
as `goodbye` with 0.91 confidence, because five of the new `goodbye` examples
contained the word "you". No metric moved, because the phrase was nowhere in
the corpus.

`tests/nlu_regression.yml` is a held-out set of realistic phrasings — the ones
that broke, plus normal task requests, plus genuinely off-topic messages. It
is never trained on.

```bash
rasa test nlu --nlu tests/nlu_regression.yml \
              --model models/<model>.tar.gz \
              --out results/regression
```

### Unit tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_actions.py -v
```

86 tests, no network: Nominatim, Open-Meteo, Frankfurter, Wikipedia and
Climatiq are all stubbed. Every API-backed action is checked three ways — a
normal answer, a transport failure, and a malformed body — because the brief
requires each action to handle failed API responses rather than crash. The
adaptive form branches are tested here, by calling `required_slots()`
directly (see the note below on why the story tests cannot do it).

### Dialogue tests

Eleven story tests in `tests/test_stories.yml` cover the adaptive form
branches, the two-stage clarification loop, trip plan → human handover,
Wikipedia → hotels, weather → activities, carbon → offset, and a
course-companion regression check.

```bash
rasa test core --stories tests/test_stories.yml
```

Outputs: `results/story_report.json`, `results/story_confusion_matrix.png`,
`results/failed_test_stories.yml`.

`rasa test core` reports **8** stories from the 11 written, and that is worth
understanding rather than papering over. It evaluates predicted *actions*;
inside a form the predicted action is always `trip_planning_form`, and which
question the form asks next is decided by `required_slots()` on the action
server — which this command never runs. The four adaptive-branch stories are
therefore identical from its point of view and get collapsed into one. They
document the branches; `tests/test_actions.py` tests them.

### Frozen results

`results/` is overwritten by every `rasa test` run, so each evaluation the
report cites is copied into `evaluation/<name>/`. See `evaluation/README.md`
for what each snapshot represents.

---

## When the bot does not understand

Three stages, each of which admits plainly that the bot did not follow — and
then says what it *can* do, so the user is not left guessing:

1. **Clarification.** `action_default_ask_affirmation` offers DIET's top two
   guesses as buttons with readable labels ("When assignments are due", not
   `ask_deadlines`), plus "Plan a sustainable trip" and "None of these".
2. **Rephrase.** `utter_ask_rephrase` lists what the bot handles and offers
   the three most common tasks as buttons.
3. **Human handover.** `action_default_fallback` escalates with the full
   conversation context, as the brief requires.

A message the bot recognises as off-topic ("what is 17 times 34", "tell me a
joke") skips the guessing and goes straight to the scope explanation, via the
`off_topic` intent. That intent is kept separate from `out_of_scope`, which
Rasa's two-stage fallback reserves for the "None of these" denial inside the
clarification loop.

`FallbackClassifier.threshold` was raised from 0.4 to 0.6 on the evidence in
`scripts/threshold_sweep.py`: across 56 held-out in-scope messages the lowest
correct prediction scores 0.91, so the higher cut-off costs nothing, while
messages the bot should not answer score far lower.

```bash
python scripts/threshold_sweep.py models/<model>.tar.gz
```

## Known limitations

- **Copenhagen and Oslo POI data missing.** Overpass returned SSL errors on
  the initial fetch. Re-run `python scripts/fetch_pois.py Copenhagen Oslo`
  when the Overpass API is stable. The action degrades gracefully with a
  friendly "run the pre-fetch script" message for missing cities.
- **`ask_place_description` intent F1 is low** (~0.12 on cross-validation).
  Its "tell me about X" phrasing overlaps `ask_about_tool`. Adding a dozen
  more disambiguating training examples would improve this considerably.
- **Ticket prices are estimates, not fares.** Amadeus was decommissioned and
  no free API returns per-route fares, so `EST_COST_EUR_PER_KM` holds average
  European costs per kilometre. Every answer that shows a price says so.
- **Carbon falls back silently in one sense only.** If Climatiq is
  unreachable, the local table answers instead — the figures stay close
  (both derive from BEIS 2023) and the reply names the source that was
  actually used, but it is an average rather than a live lookup.
- **Handover is a demo stub.** The full context package is printed to the
  actions server log; a real deployment would post it to Slack, Zendesk or
  an email queue. See `_build_handover_package()` in `actions/actions.py`.

---

## Deployment notes (HuggingFace Spaces)

The module recommends HuggingFace Spaces Docker SDK for hosting. Adapting
`docker-compose.yml` to a single-container Space is straightforward: use a
supervisord (or a tiny bash wrapper) to run `rasa run actions &` and
`rasa run --enable-api ... &` in the same container, then `streamlit run`
in the foreground. Set `RASA_URL=http://localhost:5005/...` inside that
container.

For a multi-container hosting environment (Fly.io, Railway,
Docker-in-Docker VMs), the current three-service compose file works
unchanged; expose only the `streamlit` service publicly.

---

## Repository layout

See `HANDOFF.md` for the full file map and rationale. Short version:

```
config.yml                # NLU pipeline + policies
domain.yml                # intents, entities, slots, form, responses, buttons
data/nlu.yml              # training examples
data/rules.yml            # deterministic intent → action
data/stories.yml          # multi-turn dialogue paths
data/eco_data/*.json      # pre-fetched OSM POIs
actions/actions.py        # 13 custom actions + helpers
scripts/fetch_pois.py     # Overpass pre-fetch tool
streamlit_app.py          # UI
tests/test_stories.yml    # story tests
Dockerfile.rasa           # + Dockerfile.actions, Dockerfile.streamlit
docker-compose.yml
endpoints.yml             # local dev (localhost:5055)
endpoints.docker.yml      # compose (actions:5055)
```

---

## Credit

Assignment: *Eco-Travel Advisor — Conversational Agent for Sustainable
Tourism Planning using the Rasa Platform*. Course leader: Dr. Abdelaziz
Triki. Author: Baris Alaca (MSc AI, BSBI Berlin, cohort G1).
