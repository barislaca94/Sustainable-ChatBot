# Eco-Travel Advisor

A conversational sustainable-travel planner built with **Rasa Open Source 3.6** and
**Streamlit**. The bot gathers trip details through an adaptive multi-turn form,
compares transport modes by carbon and estimated cost, ranks places to stay by
proxy sustainability signals from OpenStreetMap, and hands the conversation to a
human advisor, with its full context, when it cannot help.

Built for the MSc AI module *Advanced Conversational UI Design & Chatbot
Development* (BSBI Berlin). The report that accompanies this repository describes
the design, the evaluation and its limitations in full.

---

## Quick start (local, no Docker)

Requirements: Python 3.10 and `pip` (macOS or Linux shown; Windows works with the
usual path changes). A trained model is included in `models/`, so no training is
needed to run the bot.

```bash
# 1. Virtual environment and dependencies (exact versions are pinned)
python3.10 -m venv rasa-env
source rasa-env/bin/activate
pip install -r requirements-rasa.txt -r requirements-actions.txt -r requirements-streamlit.txt

# 2. Optional: a Climatiq key for live carbon figures (see below)
cp .env.example .env

# 3. Start the three processes, each in its own terminal
rasa run --endpoints endpoints.yml --model models/20261004-141855-sparse-octagon.tar.gz   # 5005
rasa run actions                                                                          # 5055
streamlit run streamlit_app.py                                                            # 8501

# 4. Open http://localhost:8501 and choose "Chat" in the sidebar
```

Retrain only if you change `config.yml`, `domain.yml` or anything in `data/`:
`rasa data validate && rasa train`, then pass the new model to `--model`.

## Quick start (Docker Compose)

```bash
cp .env.example .env      # required: docker-compose.yml reads it (the key inside is optional)
docker compose up --build
# then open http://localhost:8501
```

| Service     | Port | Published on the host | What it does |
|-------------|-----:|:---:|---|
| `streamlit` | 8501 | yes | Chat UI: bot-sent buttons, colour-coded cards, handover banner |
| `rasa`      | 5005 | no  | NLU + dialogue management, REST channel `/webhooks/rest/webhook` |
| `actions`   | 5055 | no  | Custom actions and external API calls |

Only the UI is reachable from outside; Rasa and the action server talk on the
internal compose network. Rasa runs **without** `--enable-api`: the chat webhook
works without it, while the HTTP API has no authentication by default and would let
anyone read a conversation or replace the model. No `--cors` is set because only the
Streamlit server, never a browser, calls Rasa.

> The Docker files were written and reviewed but not run on the development
> machine, which has no Docker installation. The action server's start command and
> pinned requirements were checked in a clean virtual environment with only
> `requirements-actions.txt` installed.

---

## Architecture

```
  Streamlit UI (8501)  -- POST /webhooks/rest/webhook -->  Rasa (5005)
  renders bot buttons,                                     NLU: WhitespaceTokenizer, RegexFeaturizer,
  colour cards,                                            LexicalSyntacticFeaturizer, CountVectors
  handover banner                                          (word + char_wb 1-4), DIETClassifier,
                                                           FallbackClassifier (0.70 / ambiguity 0.1),
                                                           SafetyGate (custom, deterministic)
                                                           Core: RulePolicy, MemoizationPolicy,
                                                           TEDPolicy, UnexpecTEDIntentPolicy
                                                                  |
                                                                  | POST /webhook
                                                                  v
                                                           Action server (5055), actions/actions.py
                                                           adaptive trip form, transport and carbon
                                                           ranking, hotels, activities, place
                                                           descriptions, weather, currency,
                                                           clarification and human handover
                                                                  |
        Nominatim (live) · Open-Meteo (live) · Frankfurter (live) · Wikipedia (live)
        Climatiq (live, local table as fallback) · Overpass (pre-fetched to data/eco_data/)
```

Main behaviours, with the course material they follow:

- **Adaptive form** (`ValidateTripPlanningForm.required_slots`): destination,
  origin, dates, budget and sustainability level are always asked; transport
  preference only after "high" sustainability, trip length only on a budget under
  500 EUR. Free-text answers are validated and re-asked when unusable.
- **Buttons generated by custom actions**: the destination question offers the
  cities with pre-fetched data; the preference questions are also asked by
  `action_ask_trip_planning_form_*` actions. Preference buttons send
  `/inform{...}` payloads, so a click skips NLU.
- **Two-stage clarification**: a low-confidence message gets the classifier's two
  best guesses as readable buttons, then a rephrase request with the bot's scope,
  then a human handover.
- **Human handover** (`action_human_handover`): a ticket with collected slots, the
  last trip plan, the last intent and its confidence, and the recent transcript.
  In this demo the package is written to the action server's log; no ticketing
  system or human advisor is connected, and the bot says so.
- **Safety**: a deterministic gate (`components/safety_gate.py`,
  `data/safety_patterns.yml`) rejects empty or over-long input and routes visa,
  health and safety questions and abuse to fixed replies. It is conservative and
  incomplete by design; the NLU intents remain a second layer.
- **Follow-up context**: a request that names no place ("and hotels?") uses the
  destination of the last completed plan and says so.

---

## External APIs

Amadeus, named in the assignment brief, was decommissioned on 17 July 2026; the
module's Eco-Travel APIs guide lists the replacements used here.

| API          | Key | Use | Called |
|--------------|:---:|---|---|
| Nominatim    | no  | geocoding | live, at most 1 request/second, identifying User-Agent |
| Overpass     | no  | hotels, transit stops, attractions | **offline**: `scripts/fetch_pois.py` writes `data/eco_data/*.json` |
| Wikipedia    | no  | place summaries | live |
| Open-Meteo   | no  | weather | live |
| Frankfurter  | no  | exchange rates | live |
| Climatiq     | yes (free tier) | carbon per transport mode | live, with a local factor table as fallback |

Overpass is never called during a conversation: retries can take far longer than
the brief's three-second target, so the data is fetched ahead of time. Set
`ECO_USER_AGENT` in `.env` to your own contact address if you run or fork the bot.

**Carbon figures.** `estimate_carbon()` sends one batched Climatiq request per
answer (2.5 s timeout, results cached per mode and distance). Without a key, or if
the call fails, the local table (`EMISSION_FACTORS`, BEIS 2023 / Our World in Data)
answers instead, and every reply names the source it used.
`python scripts/climatiq_check.py` compares the two per mode.

**Costs** are indicative per-kilometre averages, not fares: no fare source is
connected. Every answer that shows a price says so.

---

## Sustainability and ethics

- OpenStreetMap carries almost no eco-certification data, so the bot **never**
  calls a hotel certified. It ranks hotels by proxy signals (distance to rail,
  metro or tram; small scale; no car park), says they are proxies, and points to
  independent certification schemes. Missing tags never lower a score.
- Colour bands are always paired with words ("Low emission"), so colour is never
  the only carrier of meaning. A trip-plan card without a carbon figure is a
  neutral info box.
- Offset schemes are listed by name and address as starting points, not
  endorsements, with the caveat that offsets complement, not replace, low-emission
  choices.
- **Privacy**: conversation state lives in Rasa's in-memory tracker store and is
  lost on restart; nothing is written to a database. A handover writes the
  conversation summary to the action server's log, which a real deployment would
  have to treat as personal data (GDPR). The bot never asks for payment or
  passport details.

---

## Evaluation

The NLU numbers below are from the report's measurement protocol. Validation and
test numbers are kept apart: the dev sets and cross-validation were used for every
choice; the frozen final test set was run once, on 2026-10-04, after all choices
were made.

| What | Result |
|---|---|
| Paired 3×3 cross-validation, 544 examples, macro F1 (validation) | DIET 0.673 · TF-IDF word + LR 0.524 · TF-IDF word + char + LR 0.645 (gate off); DIET 0.700 with the SafetyGate |
| Final test, 92 sentences, classifier level (official) | accuracy 0.793 · macro F1 0.796 · request_human_advisor recall 6/6 · 13 of 20 intents reach F1 ≥ 0.85 |
| Final test, bot decisions at threshold 0.70 (official) | 64 of 84 in-scope requests answered, 92.2% of them correctly; 23.8% sent to clarification |

The official numbers belong to model `20261003-172117-obvious-link`. The served
model, `20261004-141855-sparse-octagon`, differs only in the dialogue layer (the
preference buttons were moved into custom actions after the final test); its NLU
component files are byte-identical. The final test files (`tests/final_test*.yml`)
are pinned by SHA-256 in `tests/test_actions.py` and must never be used for
training or tuning. Button clicks skip NLU, so these figures describe typed input.

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_actions.py -q          # 176 unit tests, no network (all APIs stubbed)
rasa test core --stories tests/test_stories.yml --model models/20261004-141855-sparse-octagon.tar.gz
rasa run actions &                       # the smoke test uses the real action server
python scripts/dialogue_smoke.py models/20261004-141855-sparse-octagon.tar.gz
```

- **Unit tests** check every API-backed action three ways (normal answer,
  transport failure, malformed body), the form's adaptive branches, the safety
  gate, and that the dev and final test sets share no sentence with the training
  data.
- **Story tests**: `tests/test_stories.yml` holds 19 stories; `rasa test core`
  reports 16, because inside a form every turn predicts the same action, so the
  adaptive-branch stories collapse into one (the branches are unit-tested
  instead). Result on the served model: 16/16 stories, 61/61 actions.
- **Smoke test**: 64 turns through `Agent.handle_text`, failing any turn that is
  slow (> 3 s), silent or wrong. One turn fails, a privacy question that goes to
  clarification (known limitation below).

Measurement scripts: `threshold_sweep.py` (fallback threshold on the dev sets),
`safety_gate_report.py`, `baseline_tfidf_lr.py`, `cv_compare.py`, `cv_epochs.py`,
`cv_min_ngram.py`, `final_test_decisions.py`; each documents its usage at the top.

---

## Known limitations

- Weak intents on the final test: `ask_currency` (currency names rather than ISO
  codes), `inform` without context, `bot_challenge` / `off_topic` on personal
  questions to the bot. These were not tuned on the test set; they are reported.
- Privacy questions phrased as questions about the bot can go to clarification
  instead of the privacy answer.
- Hotel data exists for Amsterdam, Barcelona, Berlin, Kyoto, Lisbon and Paris;
  Paris has no pre-fetched attractions. Activity ideas also cover Copenhagen and
  Oslo, which have no hotel data.
- The first question about a new route can exceed three seconds while Nominatim
  and Climatiq are called; repeated questions are served from an in-process cache.
- The handover is a demo: the package goes to a log, not to a person.
- Streamlit is a prototype UI, as the brief notes; there is no hotel carousel and
  no voice input.

---

## Data sources and licences

- `data/eco_data/*.json` contain data from **OpenStreetMap**, © OpenStreetMap
  contributors, available under the Open Database License (ODbL):
  https://www.openstreetmap.org/copyright. The extracts are distributed here under
  the same licence.
- `data/nlu_public.yml`: user messages from CLINC150 (Larson et al., 2019;
  CC BY 3.0) and MultiWOZ 2.2 (MIT licence); `data/nlu_sgd.yml`: user turns from
  the Schema-Guided Dialogue dataset (Rastogi et al., 2020; CC BY-SA 4.0, and that
  file is distributed under the same licence). Full references, DOIs and the
  changes made are in each file's header.

---

## Repository layout

```
config.yml, domain.yml        NLU pipeline and policies; intents, slots, form, responses
data/                         training data (nlu*.yml), rules, stories, safety patterns, OSM extracts
actions/actions.py            custom actions (19 classes, incl. the form validator)
components/safety_gate.py     deterministic input and safety gate (custom NLU component)
models/                       the served model
streamlit_app.py              chat UI
tests/                        unit tests, story tests, dev sets, frozen final test set
scripts/                      data fetching, measurement and smoke-test scripts
Dockerfile.*, docker-compose.yml, endpoints.docker.yml   containers
endpoints.yml, credentials.yml                           local Rasa configuration
```

## Credit

Assignment: *Eco-Travel Advisor — Conversational Agent for Sustainable Tourism
Planning using the Rasa Platform*, BSBI Berlin. Author: Baris Alaca.
