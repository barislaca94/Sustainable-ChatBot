"""Unit tests for the custom actions.

No network: every external call (Nominatim, Open-Meteo, Frankfurter,
Wikipedia, Climatiq) is replaced with a stub, so the suite runs offline and
deterministically. Each API-backed action is checked three ways — a normal
answer, a transport failure, and a malformed body — because the brief
requires every action to handle failed API responses rather than crash.

    pip install -r requirements-dev.txt
    pytest tests/test_actions.py -v
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rasa_sdk import Tracker  # noqa: E402
from rasa_sdk.executor import CollectingDispatcher  # noqa: E402

from actions import actions  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_tracker(slots: Optional[Dict[str, Any]] = None,
                 latest_message: Optional[Dict[str, Any]] = None,
                 events: Optional[List[Dict[str, Any]]] = None,
                 sender_id: str = "test-user") -> Tracker:
    """A tracker with just enough state for an action to run."""
    return Tracker(
        sender_id=sender_id,
        slots=slots or {},
        latest_message=latest_message or {},
        events=events or [],
        paused=False,
        followup_action=None,
        active_loop={},
        latest_action_name=None,
    )


def texts(dispatcher: CollectingDispatcher) -> List[str]:
    return [m.get("text", "") for m in dispatcher.messages]


class FakeResponse:
    """Stand-in for requests.Response."""

    def __init__(self, payload: Any = None, status_code: int = 200,
                 raise_json: bool = False):
        self._payload = payload
        self.status_code = status_code
        self._raise_json = raise_json

    def json(self) -> Any:
        if self._raise_json:
            raise ValueError("no JSON object could be decoded")
        return self._payload


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Geocoding backs off before retrying; tests should not wait for it."""
    monkeypatch.setattr(actions.time, "sleep", lambda seconds: None)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Fail loudly if a test forgets to stub an HTTP call."""
    def blocked(*args, **kwargs):
        raise AssertionError("unstubbed network call in a unit test")

    monkeypatch.setattr(actions.requests, "get", blocked)
    monkeypatch.setattr(actions.requests, "post", blocked)
    # Geocoding is cached in-process; keep tests independent of each other.
    actions._GEOCODE_CACHE.clear()
    actions._CARBON_CACHE.clear()


@pytest.fixture
def no_climatiq_key(monkeypatch):
    monkeypatch.setattr(actions, "climatiq_api_key", lambda: None)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_haversine_matches_known_distance():
    # London to Paris is about 344 km great-circle.
    km = actions._haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330 < km < 360


@pytest.mark.parametrize("kg,expected", [(10, "🟢"), (100, "🟡"), (400, "🔴")])
def test_carbon_to_band_uses_absolute_size(kg, expected):
    assert actions.carbon_to_band(kg) == expected


def test_intensity_band_separates_modes_not_distances():
    # A long train trip and a short one are both low-intensity; a flight is
    # high-intensity at either length. This is the bug the band fix addressed.
    assert actions.intensity_band(31, 1000) == "🟢"
    assert actions.intensity_band(310, 10000) == "🟢"
    assert actions.intensity_band(126, 1000) == "🔴"


def test_intensity_band_falls_back_when_distance_unknown():
    assert actions.intensity_band(10, 0) == actions.carbon_to_band(10)


def test_signals_score_grades_rail_distance():
    near, band_near = actions.signals_score({"nearest_rail_m": 100})
    mid, _ = actions.signals_score({"nearest_rail_m": 600})
    far, _ = actions.signals_score({"nearest_rail_m": 1200})
    assert near > mid > far
    assert band_near == "🟡"  # one signal alone is not enough for green


def test_signals_score_missing_tags_never_subtract():
    # No OSM evidence at all is shown as "not enough data" (neutral), never
    # as a negative judgement on the hotel (A10).
    score, band = actions.signals_score({})
    assert score == 0.0
    assert band == "⚪"


def test_signals_score_weak_but_present_evidence_is_red():
    score, band = actions.signals_score({"nearest_stop_m": 100})
    assert 0 < score < 0.8
    assert band == "🔴"


def test_signals_score_green_needs_two_signals():
    score, band = actions.signals_score({"nearest_rail_m": 100, "rooms": 12})
    assert score >= 1.5
    assert band == "🟢"


def test_is_rail_stop_rejects_bus_stops():
    assert actions._is_rail_stop({"tags": {"railway": "station"}})
    assert actions._is_rail_stop({"tags": {"subway": "yes"}})
    assert not actions._is_rail_stop({"tags": {"bus": "yes"}})


def test_format_signals_reports_accessibility():
    text = actions.format_signals({"nearest_rail_m": 120, "wheelchair": "yes"})
    assert "120m to rail" in text
    assert "step-free access: yes" in text


def test_format_signals_says_so_when_nothing_is_known():
    assert "no usable OSM tags" in actions.format_signals({})


# --------------------------------------------------------------------------
# Geocoding: "we cannot reach the service" must not become "no such place"
# --------------------------------------------------------------------------

NOMINATIM_BERLIN = [{"lat": "52.52", "lon": "13.40"}]


def test_geocode_returns_coordinates(monkeypatch):
    monkeypatch.setattr(
        actions.requests, "get", lambda *a, **k: FakeResponse(NOMINATIM_BERLIN)
    )
    coords, status = actions.geocode_city_result("Berlin")
    assert coords == (52.52, 13.40)
    assert status == actions.GEOCODE_OK


def test_geocode_rate_limited_is_unavailable_not_missing(monkeypatch):
    # A 429 body is plain text, so .json() raises. This used to escape the
    # action entirely and, via validate_origin, told the user that Berlin
    # does not exist.
    monkeypatch.setattr(
        actions.requests, "get",
        lambda *a, **k: FakeResponse(status_code=429, raise_json=True),
    )
    coords, status = actions.geocode_city_result("Berlin")
    assert coords is None
    assert status == actions.GEOCODE_UNAVAILABLE


def test_geocode_retries_once_then_succeeds(monkeypatch):
    answers = [FakeResponse(status_code=503, raise_json=True),
               FakeResponse(NOMINATIM_BERLIN)]
    monkeypatch.setattr(actions.requests, "get", lambda *a, **k: answers.pop(0))
    coords, status = actions.geocode_city_result("Berlin")
    assert status == actions.GEOCODE_OK
    assert coords == (52.52, 13.40)
    assert answers == []


def test_geocode_html_body_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        actions.requests, "get", lambda *a, **k: FakeResponse(raise_json=True)
    )
    assert actions.geocode_city_result("Berlin")[1] == actions.GEOCODE_UNAVAILABLE


def test_geocode_empty_result_is_not_found(monkeypatch):
    monkeypatch.setattr(actions.requests, "get", lambda *a, **k: FakeResponse([]))
    coords, status = actions.geocode_city_result("Xyzzyville")
    assert coords is None
    assert status == actions.GEOCODE_NOT_FOUND


def test_geocode_network_failure_is_unavailable(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(actions.requests, "get", boom)
    assert actions.geocode_city_result("Berlin")[1] == actions.GEOCODE_UNAVAILABLE


def test_geocode_caches_successful_lookups(monkeypatch):
    calls = []

    def once(*args, **kwargs):
        calls.append(1)
        return FakeResponse(NOMINATIM_BERLIN)

    monkeypatch.setattr(actions.requests, "get", once)
    actions.geocode_city_result("Berlin")
    actions.geocode_city_result("berlin")
    assert len(calls) == 1


def test_distance_reports_why_it_is_zero(monkeypatch):
    monkeypatch.setattr(
        actions, "geocode_city_result",
        lambda city: (None, actions.GEOCODE_UNAVAILABLE),
    )
    km, status = actions.approx_distance_result("Berlin", "Paris")
    assert km == 0.0
    assert status == actions.GEOCODE_UNAVAILABLE


def test_transport_action_says_the_map_service_is_down(monkeypatch):
    monkeypatch.setattr(
        actions, "approx_distance_result",
        lambda o, d: (0.0, actions.GEOCODE_UNAVAILABLE),
    )
    dispatcher = CollectingDispatcher()
    actions.ActionSuggestTransport().run(
        dispatcher, make_tracker({"origin": "Berlin", "destination": "Paris"}), {}
    )
    message = texts(dispatcher)[0]
    assert "couldn't reach the map service" in message
    assert "Berlin" not in message  # never implies the place does not exist


def test_transport_action_still_reports_a_genuinely_unknown_place(monkeypatch):
    monkeypatch.setattr(
        actions, "approx_distance_result",
        lambda o, d: (0.0, actions.GEOCODE_NOT_FOUND),
    )
    dispatcher = CollectingDispatcher()
    actions.ActionSuggestTransport().run(
        dispatcher, make_tracker({"origin": "Xyzzy", "destination": "Paris"}), {}
    )
    assert "couldn't locate" in texts(dispatcher)[0]


# --------------------------------------------------------------------------
# Hotel ranking
# --------------------------------------------------------------------------

@pytest.fixture
def city_cache(tmp_path, monkeypatch):
    """A two-hotel city: one next to a metro, one far from anything."""
    hotels = {
        "city": "Testville",
        "centre": {"lat": 0.0, "lon": 0.0},
        "records": [
            {"id": "n/1", "name": "Near Station Inn", "lat": 0.0, "lon": 0.0,
             "tags": {"tourism": "hotel", "rooms": "20"}},
            {"id": "n/2", "name": "Grand Palace Hotel", "lat": 0.2, "lon": 0.2,
             "tags": {"tourism": "hotel", "stars": "5"}},
            {"id": "n/3", "name": None, "lat": 0.0, "lon": 0.0, "tags": {}},
        ],
    }
    transit = {
        "city": "Testville",
        "records": [
            {"id": "n/9", "name": "Central", "lat": 0.0005, "lon": 0.0,
             "tags": {"railway": "station"}},
        ],
    }
    attractions = {
        "city": "Testville",
        "centre": {"lat": 0.0, "lon": 0.0},
        "records": [
            {"id": "n/5", "name": "City Museum", "lat": 0.001, "lon": 0.0,
             "tags": {"tourism": "museum"}},
            {"id": "n/6", "name": "Vague Viewpoint", "lat": 0.002, "lon": 0.0,
             "tags": {"tourism": "attraction"}},
        ],
    }
    (tmp_path / "testville_hotels.json").write_text(json.dumps(hotels))
    (tmp_path / "testville_transit.json").write_text(json.dumps(transit))
    (tmp_path / "testville_attractions.json").write_text(json.dumps(attractions))
    monkeypatch.setattr(actions, "ECO_DATA_DIR", tmp_path)
    return tmp_path


def test_rank_eco_hotels_orders_by_signals(city_cache):
    ranked = actions.rank_eco_hotels("Testville")
    assert [h["name"] for h in ranked] == ["Near Station Inn", "Grand Palace Hotel"]
    assert ranked[0]["band"] == "🟢"


def test_rank_eco_hotels_skips_unnamed_records(city_cache):
    assert all(h["name"] for h in actions.rank_eco_hotels("Testville"))


def test_rank_eco_hotels_demotes_expensive_stars_on_a_tight_budget(city_cache):
    ranked = actions.rank_eco_hotels("Testville", budget_per_night=50)
    palace = next(h for h in ranked if h["name"] == "Grand Palace Hotel")
    assert palace["note"] is not None
    assert "5★" in palace["note"]


def test_rank_eco_hotels_unknown_city_returns_empty(city_cache):
    assert actions.rank_eco_hotels("Atlantis") == []


def test_nearby_cultural_sites_drops_generic_attractions(city_cache):
    sites = actions.nearby_cultural_sites("Testville")
    assert [s["name"] for s in sites] == ["City Museum"]


# --------------------------------------------------------------------------
# Carbon estimation
# --------------------------------------------------------------------------

def test_estimate_carbon_uses_local_table_without_a_key(no_climatiq_key):
    result, label = actions.estimate_carbon(["train", "flight"], 1000)
    assert result["train"] == pytest.approx(1000 * actions.EMISSION_FACTORS["train"])
    assert "no Climatiq key" in label


def test_estimate_carbon_reads_live_answer(monkeypatch):
    monkeypatch.setattr(actions, "climatiq_api_key", lambda: "test-key")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return FakeResponse({"results": [{"co2e": 12.5, "co2e_unit": "kg"}]})

    monkeypatch.setattr(actions.requests, "post", fake_post)
    result, label = actions.estimate_carbon(["train"], 400)

    assert result["train"] == 12.5
    assert "Climatiq live estimate" in label
    assert captured["url"] == actions.CLIMATIQ_BATCH_URL
    assert captured["body"][0]["parameters"]["distance"] == 400


def test_estimate_carbon_converts_tonnes(monkeypatch):
    monkeypatch.setattr(actions, "climatiq_api_key", lambda: "test-key")
    monkeypatch.setattr(
        actions.requests, "post",
        lambda *a, **k: FakeResponse({"results": [{"co2e": 1.2, "co2e_unit": "t"}]}),
    )
    result, _ = actions.estimate_carbon(["flight"], 5000)
    assert result["flight"] == pytest.approx(1200.0)


def test_estimate_carbon_falls_back_on_timeout(monkeypatch):
    monkeypatch.setattr(actions, "climatiq_api_key", lambda: "test-key")

    def timeout(*args, **kwargs):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(actions.requests, "post", timeout)
    result, label = actions.estimate_carbon(["train"], 1000)

    assert result["train"] == pytest.approx(1000 * actions.EMISSION_FACTORS["train"])
    assert "Climatiq unavailable" in label


def test_estimate_carbon_falls_back_on_malformed_body(monkeypatch):
    monkeypatch.setattr(actions, "climatiq_api_key", lambda: "test-key")
    monkeypatch.setattr(
        actions.requests, "post",
        lambda *a, **k: FakeResponse({"unexpected": "shape"}),
    )
    result, label = actions.estimate_carbon(["train"], 1000)
    assert result["train"] == pytest.approx(1000 * actions.EMISSION_FACTORS["train"])
    assert "Climatiq unavailable" in label


def test_estimate_carbon_caches_repeat_lookups(monkeypatch):
    monkeypatch.setattr(actions, "climatiq_api_key", lambda: "test-key")
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse({"results": [{"co2e": 5.0, "co2e_unit": "kg"}]})

    monkeypatch.setattr(actions.requests, "post", fake_post)
    actions.estimate_carbon(["train"], 250)
    actions.estimate_carbon(["train"], 250)
    assert len(calls) == 1


def test_climatiq_activity_id_switches_at_the_haul_boundary():
    short = actions.climatiq_activity_id("flight", 500)
    long_haul = actions.climatiq_activity_id("flight", 9000)
    assert "short_haul" in short
    assert "long_haul" in long_haul
    assert actions.climatiq_activity_id("hot air balloon", 100) is None


# --------------------------------------------------------------------------
# Weighted transport scoring
# --------------------------------------------------------------------------

def test_score_transport_options_ranks_rail_first_for_high_sustainability(no_climatiq_key):
    options, excluded, _ = actions.score_transport_options(
        1000, sustainability_level="high"
    )
    assert options[0]["mode"] in ("train", "bus")
    assert options[-1]["mode"] in ("car", "flight")
    assert excluded == []


def test_score_transport_options_weights_shift_with_the_stated_level(no_climatiq_key):
    # Train is the cleanest option and the dearest; bus is cheapest and close
    # behind on carbon. Saying sustainability matters must narrow the gap
    # between them, since the axis the train wins on now counts for more.
    def gap(level: str) -> float:
        options, _, _ = actions.score_transport_options(
            1000, sustainability_level=level
        )
        by_mode = {o["mode"]: o["score"] for o in options}
        return by_mode["train"] - by_mode["bus"]

    assert gap("high") > gap("medium") > gap("low")


def test_score_transport_options_puts_the_cheapest_first_for_a_price_first_user(
        no_climatiq_key):
    options, _, _ = actions.score_transport_options(1000, sustainability_level="low")
    cheapest = min(options, key=lambda o: o["cost_eur"])["mode"]
    assert options[0]["mode"] == cheapest


def test_score_transport_options_honours_train_or_bus(no_climatiq_key):
    options, excluded, _ = actions.score_transport_options(
        800, transport_preference="train_or_bus"
    )
    assert {o["mode"] for o in options} == {"train", "bus"}
    assert set(excluded) == {"car", "flight"}


def test_score_transport_options_any_low_carbon_drops_flying(no_climatiq_key):
    options, excluded, _ = actions.score_transport_options(
        800, transport_preference="any_low_carbon"
    )
    assert "flight" not in {o["mode"] for o in options}
    assert excluded == ["flight"]


def test_score_transport_options_warns_when_travel_eats_the_budget(no_climatiq_key):
    options, _, _ = actions.score_transport_options(
        2000, budget="300", transport_preference="train_or_bus"
    )
    assert any(o["warning"] for o in options)


def test_score_transport_options_ignores_an_unusable_budget(no_climatiq_key):
    options, _, _ = actions.score_transport_options(500, budget="lots")
    assert all(o["warning"] is None for o in options)


def test_nightly_budget_needs_both_budget_and_length():
    assert actions.nightly_budget(make_tracker({"budget": "700"})) is None
    assert actions.nightly_budget(make_tracker({"trip_length": "week"})) is None
    per_night = actions.nightly_budget(
        make_tracker({"budget": "700", "trip_length": "week"})
    )
    assert per_night == pytest.approx(700 * 0.5 / 7)


# --------------------------------------------------------------------------
# API-backed actions
# --------------------------------------------------------------------------

WEATHER_OK = {
    "current": {
        "temperature_2m": 18.0,
        "wind_speed_10m": 12.0,
        "relative_humidity_2m": 55,
        "weather_code": 0,
    },
    "daily": {"temperature_2m_max": [19.0, 21.0], "precipitation_sum": [0.0, 0.0]},
}


def run_weather(monkeypatch, weather_payload, geocode=(52.5, 13.4),
                geocode_status=None, raise_request=False,
                raise_json=False) -> List[str]:
    status = geocode_status or (
        actions.GEOCODE_OK if geocode else actions.GEOCODE_NOT_FOUND
    )
    monkeypatch.setattr(
        actions, "geocode_city_result", lambda city: (geocode, status)
    )

    def fake_get(url, params=None, headers=None, timeout=None):
        if raise_request:
            raise requests.ConnectionError("down")
        return FakeResponse(weather_payload, raise_json=raise_json)

    monkeypatch.setattr(actions.requests, "get", fake_get)
    dispatcher = CollectingDispatcher()
    actions.ActionGetWeather().run(
        dispatcher, make_tracker({"city_name": "Berlin"}), {}
    )
    return texts(dispatcher)


def test_weather_reports_conditions(monkeypatch):
    assert "18.0 C" in run_weather(monkeypatch, WEATHER_OK)[0]


def test_weather_handles_unreachable_service(monkeypatch):
    assert "could not reach" in run_weather(
        monkeypatch, None, raise_request=True
    )[0].lower()


def test_weather_handles_malformed_body(monkeypatch):
    assert "unexpected" in run_weather(monkeypatch, {"error": True})[0].lower()


def test_weather_handles_non_json_body(monkeypatch):
    assert "could not reach" in run_weather(
        monkeypatch, None, raise_json=True
    )[0].lower()


def test_weather_handles_unknown_place(monkeypatch):
    assert "could not find" in run_weather(
        monkeypatch, WEATHER_OK, geocode=None
    )[0].lower()


def test_weather_separates_a_down_map_service_from_an_unknown_place(monkeypatch):
    message = run_weather(
        monkeypatch, WEATHER_OK, geocode=None,
        geocode_status=actions.GEOCODE_UNAVAILABLE,
    )[0]
    assert "couldn't reach the map service" in message
    assert "could not find" not in message.lower()


def test_currency_reports_rate(monkeypatch):
    monkeypatch.setattr(
        actions.requests, "get",
        lambda *a, **k: FakeResponse({"rates": {"EUR": 0.92}, "date": "2026-09-16"}),
    )
    dispatcher = CollectingDispatcher()
    actions.ActionGetCurrency().run(
        dispatcher, make_tracker({"from_currency": "usd", "to_currency": "eur"}), {}
    )
    assert "1 USD = 0.92 EUR" in texts(dispatcher)[0]


def test_currency_handles_missing_pair(monkeypatch):
    monkeypatch.setattr(
        actions.requests, "get", lambda *a, **k: FakeResponse({"rates": {}})
    )
    dispatcher = CollectingDispatcher()
    actions.ActionGetCurrency().run(
        dispatcher, make_tracker({"from_currency": "usd", "to_currency": "xyz"}), {}
    )
    assert "could not fetch" in texts(dispatcher)[0].lower()


def test_currency_handles_transport_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(actions.requests, "get", boom)
    dispatcher = CollectingDispatcher()
    actions.ActionGetCurrency().run(
        dispatcher, make_tracker({"from_currency": "usd", "to_currency": "eur"}), {}
    )
    assert "could not reach" in texts(dispatcher)[0].lower()


def test_describe_place_returns_summary(monkeypatch, city_cache):
    monkeypatch.setattr(
        actions.requests, "get",
        lambda *a, **k: FakeResponse({
            "title": "Testville",
            "extract": "Testville is a small town.",
            "content_urls": {"desktop": {"page": "https://example.org/Testville"}},
        }),
    )
    dispatcher = CollectingDispatcher()
    events = actions.ActionDescribePlace().run(
        dispatcher, make_tracker({"destination": "Testville"}), {}
    )
    assert "small town" in texts(dispatcher)[0]
    assert any(e.get("value") == "Testville" for e in events)


def test_describe_place_handles_disambiguation(monkeypatch):
    monkeypatch.setattr(
        actions.requests, "get",
        lambda *a, **k: FakeResponse({"type": "disambiguation"}),
    )
    dispatcher = CollectingDispatcher()
    actions.ActionDescribePlace().run(
        dispatcher, make_tracker({"destination": "Springfield"}), {}
    )
    assert "does not have a clear summary" in texts(dispatcher)[0]


def test_describe_place_handles_unreachable_wikipedia(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(actions.requests, "get", boom)
    dispatcher = CollectingDispatcher()
    actions.ActionDescribePlace().run(
        dispatcher, make_tracker({"destination": "Kyoto"}), {}
    )
    assert "could not reach Wikipedia" in texts(dispatcher)[0]


# --------------------------------------------------------------------------
# Handover packaging
# --------------------------------------------------------------------------

def test_handover_package_contains_the_conversation():
    tracker = make_tracker(
        slots={"destination": "Kyoto", "handover_active": False},
        latest_message={"intent": {"name": "request_human_advisor", "confidence": 0.98}},
        events=[
            {"event": "user", "text": "hotels in Kyoto"},
            {"event": "bot", "text": "Here are some hotels"},
            {"event": "user", "text": "talk to a human"},
        ],
    )
    package = actions._build_handover_package(tracker)

    assert package["ticket_id"].startswith("TR-")
    assert package["last_intent"] == "request_human_advisor"
    assert package["turn_count"] == 2
    assert package["collected_slots"] == {"destination": "Kyoto"}
    assert package["transcript"][0].startswith("USER: hotels in Kyoto")


def test_handover_transcript_keeps_the_last_user_turns_not_messages():
    # 12 user turns, each answered by three bot messages: a message limit would
    # keep only bot text; the turn limit keeps the last 10 user turns (A8).
    events = []
    for i in range(12):
        events.append({"event": "user", "text": f"question {i}"})
        events += [{"event": "bot", "text": f"answer {i} part {k}"} for k in range(3)]
    transcript = actions._build_transcript(make_tracker(events=events))

    users = [line for line in transcript if line.startswith("USER: ")]
    assert users[0] == "USER: question 2" and users[-1] == "USER: question 11"
    assert len(users) == 10
    assert transcript[-1] == "BOT : answer 11 part 0 | answer 11 part 1 | answer 11 part 2"


def test_handover_package_recovers_the_cleared_trip_plan():
    summary = {"destination": "Kyoto", "recommended_mode": "train"}
    tracker = make_tracker(slots={"last_trip_summary": json.dumps(summary)})
    package = actions._build_handover_package(tracker)

    assert package["last_trip_plan"] == summary
    # It travels in its own field, not as a raw JSON blob among the slots.
    assert "last_trip_summary" not in package["collected_slots"]


def test_handover_package_survives_an_unparseable_summary():
    tracker = make_tracker(slots={"last_trip_summary": "not json"})
    assert actions._build_handover_package(tracker)["last_trip_plan"] == "not json"


# --------------------------------------------------------------------------
# Fallback affirmation
# --------------------------------------------------------------------------

def test_affirmation_buttons_use_readable_labels():
    tracker = make_tracker(latest_message={
        "intent": {"name": "nlu_fallback"},
        "intent_ranking": [
            {"name": "nlu_fallback", "confidence": 0.4},
            {"name": "ask_carbon_offset", "confidence": 0.3},
            {"name": "ask_weather", "confidence": 0.2},
            {"name": "greet", "confidence": 0.1},
        ],
    })
    dispatcher = CollectingDispatcher()
    actions.ActionDefaultAskAffirmation().run(dispatcher, tracker, {})
    buttons = dispatcher.messages[0]["buttons"]

    assert [b["title"] for b in buttons] == [
        "Carbon offset programmes", "The weather somewhere",
        "Plan a sustainable trip", "None of these",
    ]
    assert [b["payload"] for b in buttons] == [
        "/ask_carbon_offset", "/ask_weather", "/plan_trip", "/out_of_scope",
    ]


def test_affirmation_offers_an_escape_hatch_without_candidates():
    dispatcher = CollectingDispatcher()
    actions.ActionDefaultAskAffirmation().run(dispatcher, make_tracker(), {})
    buttons = dispatcher.messages[0]["buttons"]
    assert [b["payload"] for b in buttons] == ["/plan_trip", "/out_of_scope"]


def test_affirmation_says_what_the_bot_is_for():
    # Admitting it did not understand is not enough if the user has no idea
    # what the bot does; the scope sentence goes out with the guesses.
    dispatcher = CollectingDispatcher()
    actions.ActionDefaultAskAffirmation().run(dispatcher, make_tracker(), {})
    message = dispatcher.messages[0]["text"]
    assert "not sure what you meant" in message
    assert actions.SCOPE_SUMMARY in message


# --------------------------------------------------------------------------
# Form: adaptive questions and validation
# --------------------------------------------------------------------------

def required_slots_for(slots: Dict[str, Any]) -> List[str]:
    form = actions.ValidateTripPlanningForm()
    declared = [
        "destination", "origin", "travel_dates", "budget",
        "sustainability_level", "transport_preference", "trip_length",
    ]
    return asyncio.run(form.required_slots(
        declared, CollectingDispatcher(), make_tracker(slots), {}
    ))


def test_form_skips_transport_preference_unless_sustainability_is_high():
    assert "transport_preference" not in required_slots_for(
        {"sustainability_level": "low", "budget": "900"}
    )
    assert "transport_preference" in required_slots_for(
        {"sustainability_level": "high", "budget": "900"}
    )


def test_form_asks_trip_length_only_on_a_tight_budget():
    assert "trip_length" in required_slots_for({"budget": "300"})
    assert "trip_length" not in required_slots_for({"budget": "900"})


def test_form_asks_both_extras_when_both_apply():
    slots = required_slots_for({"sustainability_level": "high", "budget": "300"})
    assert {"transport_preference", "trip_length"} <= set(slots)


def test_required_slots_does_not_mutate_the_list_it_is_given():
    form = actions.ValidateTripPlanningForm()
    declared = ["destination", "transport_preference"]
    asyncio.run(form.required_slots(
        declared, CollectingDispatcher(),
        make_tracker({"sustainability_level": "low"}), {},
    ))
    assert declared == ["destination", "transport_preference"]


def validate(slot: str, value: Any, slots: Optional[Dict[str, Any]] = None):
    form = actions.ValidateTripPlanningForm()
    dispatcher = CollectingDispatcher()
    result = getattr(form, f"validate_{slot}")(
        value, dispatcher, make_tracker(slots), {}
    )
    return result, texts(dispatcher)


@pytest.mark.parametrize("value,expected", [
    ("800", "800"), ("around 1200 euros", "1200"),
])
def test_validate_budget_accepts_numbers(value, expected):
    result, _ = validate("budget", value)
    assert result["budget"] == expected


@pytest.mark.parametrize("value", ["soon", "", "0"])
def test_validate_budget_rejects_unusable_answers(value):
    result, messages = validate("budget", value)
    assert result["budget"] is None
    assert messages  # the user is told why


def test_validate_travel_dates_rejects_a_bare_number():
    result, messages = validate("travel_dates", "800")
    assert result["travel_dates"] is None
    assert "not a date" in messages[0]


def test_validate_travel_dates_accepts_a_natural_phrase():
    assert validate("travel_dates", "next week")[0]["travel_dates"] == "next week"


@pytest.mark.parametrize("value,expected", [
    ("very green please", "high"),
    ("moderate", "medium"),
    ("as cheap as possible", "low"),
])
def test_validate_sustainability_level_maps_phrasing(value, expected):
    assert validate("sustainability_level", value)[0]["sustainability_level"] == expected


def test_validate_sustainability_level_rejects_nonsense():
    result, messages = validate("sustainability_level", "banana")
    assert result["sustainability_level"] is None
    assert "low, medium, or high" in messages[0]


def test_validate_destination_rejects_a_number():
    result, messages = validate("destination", "12345")
    assert result["destination"] is None
    assert messages


def test_validate_origin_rejects_a_place_that_cannot_be_found(monkeypatch):
    monkeypatch.setattr(
        actions, "geocode_city_result",
        lambda city: (None, actions.GEOCODE_NOT_FOUND),
    )
    result, messages = validate("origin", "Xyzzyville")
    assert result["origin"] is None
    assert "couldn't find" in messages[0]


def test_validate_origin_accepts_a_geocodable_place(monkeypatch):
    monkeypatch.setattr(
        actions, "geocode_city_result",
        lambda city: ((51.5, -0.12), actions.GEOCODE_OK),
    )
    result, messages = validate("origin", "London")
    assert result["origin"] == "London"
    assert messages == []


def test_validate_origin_accepts_the_answer_when_the_map_service_is_down(monkeypatch):
    # Rejecting here would loop the form on a question the user answered
    # perfectly well, which is what made a transient Nominatim failure look
    # like "Berlin does not exist".
    monkeypatch.setattr(
        actions, "geocode_city_result",
        lambda city: (None, actions.GEOCODE_UNAVAILABLE),
    )
    # Hamburg has no pre-fetched data, so the map service is actually consulted.
    result, messages = validate("origin", "Hamburg")
    assert result["origin"] == "Hamburg"
    assert "can't reach the map service" in messages[0]


def validate_with_intent(slot: str, value: str, intent: str):
    form = actions.ValidateTripPlanningForm()
    dispatcher = CollectingDispatcher()
    tracker = make_tracker(latest_message={"intent": {"name": intent}})
    result = getattr(form, f"validate_{slot}")(value, dispatcher, tracker, {})
    return result, texts(dispatcher)


def test_validate_destination_rejects_an_off_topic_question():
    # The destination slot is filled from the raw text, so "What is 2+2"
    # used to become the destination and the form moved on.
    result, messages = validate_with_intent("destination", "What is 2+2", "off_topic")
    assert result["destination"] is None
    assert "Where would you like to travel?" in messages[0]


def test_validate_origin_rejects_a_question_about_the_bot():
    result, messages = validate_with_intent("origin", "what can you do", "bot_challenge")
    assert result["origin"] is None
    assert "Where will you be travelling from?" in messages[0]


def test_validate_destination_accepts_a_cached_city_without_calling_the_map(monkeypatch):
    def must_not_geocode(city):
        raise AssertionError("known cities should not hit Nominatim")

    monkeypatch.setattr(actions, "geocode_city_result", must_not_geocode)
    assert validate("destination", "Barcelona")[0]["destination"] == "Barcelona"


def test_validate_destination_rejects_an_unknown_place(monkeypatch):
    monkeypatch.setattr(
        actions, "geocode_city_result",
        lambda city: (None, actions.GEOCODE_NOT_FOUND),
    )
    result, messages = validate("destination", "Xyzzyville")
    assert result["destination"] is None
    assert "couldn't find" in messages[0]


# --------------------------------------------------------------------------
# Core fallback: understood, but no rule says what to do
# --------------------------------------------------------------------------

def test_core_fallback_explains_scope_and_reverts_the_message():
    dispatcher = CollectingDispatcher()
    events = actions.ActionCoreFallback().run(dispatcher, make_tracker(), {})

    message = dispatcher.messages[0]
    assert actions.SCOPE_SUMMARY in message["text"]
    assert any(b["payload"] == "/plan_trip" for b in message["buttons"])
    # Reverting is what stops the policies being asked the same question
    # again; the old two-stage loop in this position froze the server.
    assert [e.get("event") for e in events] == ["rewind"]


def test_validate_transport_preference_maps_button_and_typed_answers():
    assert validate("transport_preference", "train or bus only")[0][
        "transport_preference"] == "train_or_bus"
    assert validate("transport_preference", "any_low_carbon")[0][
        "transport_preference"] == "any_low_carbon"


def test_validate_trip_length_maps_phrasing():
    assert validate("trip_length", "just a weekend")[0]["trip_length"] == "weekend"
    assert validate("trip_length", "one week")[0]["trip_length"] == "week"
    assert validate("trip_length", "a longer trip")[0]["trip_length"] == "extended"


# --------------------------------------------------------------------------
# Measurement integrity: the regression set must stay held out
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _nlu_examples(path: Path) -> List[tuple]:
    """(intent, text, template) for every example in an NLU file.

    `text` drops entity markup, `template` replaces each entity with its type,
    both lowercased with punctuation and extra spaces removed — so
    "where should I stay in [Lisbon](destination)?" and "Where should I stay
    in Lisbon" compare equal, and "weather in [Oslo](city_name)" shares a
    template with "weather in [Berlin](city_name)".
    """
    import re
    import yaml

    def clean(s: str) -> str:
        s = re.sub(r"[^\w\s<>]", "", s.lower())
        return re.sub(r"\s+", " ", s).strip()

    out = []
    for block in yaml.safe_load(path.read_text())["nlu"]:
        if "intent" not in block:
            continue
        for line in block["examples"].splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            raw = line[2:]
            text = clean(re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw))
            template = clean(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2>", raw))
            out.append((block["intent"], text, template))
    return out


def test_regression_set_is_disjoint_from_training():
    # The Phase 5 "held-out" set had 23 of 59 sentences copied from the
    # training data and 7 more that only swapped an entity value, which made
    # its 0.949 meaningless. This keeps that from happening again.
    training = _nlu_examples(REPO_ROOT / "data" / "nlu.yml")
    held_out = _nlu_examples(REPO_ROOT / "tests" / "nlu_regression.yml")

    training_texts = {text for _, text, _ in training}
    training_templates = {tpl for _, _, tpl in training if "<" in tpl}

    copied = [text for _, text, _ in held_out if text in training_texts]
    templated = [text for _, text, tpl in held_out
                 if text not in training_texts and tpl in training_templates]
    assert copied == [], f"regression sentences also in data/nlu.yml: {copied}"
    assert templated == [], f"regression sentences that only swap an entity: {templated}"


# The final test set (tests/final_test*.yml) was written without access to the
# training data, frozen on 2026-10-02 and is run once, on the final model.
# These hashes make any later edit visible: change the file only to remove
# contamination, never to improve a number, and record why.
FINAL_TEST_SHA256 = {
    "final_test.yml": "c2894dd0f2e069041759957856e20a755c05d90ca794ef914f2b798345d357ab",
    "final_test_informal.yml": "fed573edb7d06a7aadf3bf8d94a1c869de8ecde55a7f9017a97d5c1f7a64bcc1",
    "final_test_negation.yml": "dcf8f89197b2b6c8f196e43132ccf0cf4eb90d1d7950d0d521b49d8e28384424",
    "final_test_typo.yml": "0cb8f2796a426567a7383e6a787d1d17febe4273c8b90bb2dd6b7d01a481e969",
}


@pytest.mark.parametrize("name", sorted(FINAL_TEST_SHA256))
def test_final_test_set_is_frozen(name):
    import hashlib

    data = (REPO_ROOT / "tests" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == FINAL_TEST_SHA256[name]


@pytest.mark.parametrize("name", sorted(FINAL_TEST_SHA256))
def test_final_test_set_is_disjoint_from_training_and_dev(name):
    final = _nlu_examples(REPO_ROOT / "tests" / name)
    for other in ("data/nlu.yml", "tests/nlu_regression.yml", "tests/offtopic_probe.yml"):
        seen = _nlu_examples(REPO_ROOT / other)
        texts = {text for _, text, _ in seen}
        templates = {tpl for _, _, tpl in seen if "<" in tpl}
        copied = [t for _, t, _ in final if t in texts]
        templated = [t for _, t, tpl in final if t not in texts and tpl in templates]
        assert copied == [], f"{name}: sentences also in {other}: {copied}"
        assert templated == [], f"{name}: sentences that only swap an entity of {other}: {templated}"


def test_probe_set_is_disjoint_from_training():
    # tests/offtopic_probe.yml is a dev set for the off-topic and safety
    # classes (Adim 2). Thresholds are tuned on it, so it must never share a
    # sentence with the training data.
    training = _nlu_examples(REPO_ROOT / "data" / "nlu.yml")
    probe = _nlu_examples(REPO_ROOT / "tests" / "offtopic_probe.yml")
    texts = {text for _, text, _ in training}
    templates = {tpl for _, _, tpl in training if "<" in tpl}
    assert [t for _, t, _ in probe if t in texts] == []
    assert [t for _, t, tpl in probe if t not in texts and tpl in templates] == []


# --------------------------------------------------------------------------
# SafetyGate (components/safety_gate.py): input validation + regex gate
# --------------------------------------------------------------------------
# The unit tests use made-up patterns, so they check the gate's logic, not the
# keyword list. The real list (data/safety_patterns.yml) is checked below only
# for structure: every category must point at an intent the domain declares.

def _gate(tmp_path, max_chars=50):
    from components.safety_gate import SafetyGate

    patterns = tmp_path / "patterns.yml"
    patterns.write_text(
        "categories:\n"
        "  - name: first\n    intent: intent_a\n    patterns: ['\\bzebra\\b']\n"
        "  - name: second\n    intent: intent_b\n    patterns: ['zebra', 'llama']\n"
    )
    return SafetyGate({"patterns_file": str(patterns), "max_chars": max_chars})


def _message(text, intent="greet", confidence=0.9):
    from rasa.shared.nlu.training_data.message import Message

    return Message(data={
        "text": text,
        "intent": {"name": intent, "confidence": confidence},
        "intent_ranking": [{"name": intent, "confidence": confidence},
                           {"name": "intent_a", "confidence": 0.05}],
    })


@pytest.mark.parametrize("text, reason", [("", "empty_input"), ("   ", "empty_input"),
                                          ("x" * 51, "too_long")])
def test_safety_gate_rejects_empty_and_overlong_input(tmp_path, text, reason):
    [msg] = _gate(tmp_path).process([_message(text)])
    assert msg.get("intent") == {"name": "invalid_input", "confidence": 1.0}
    assert msg.get("safety_gate") == reason


def test_safety_gate_overrides_the_model_on_a_pattern_match(tmp_path):
    [msg] = _gate(tmp_path).process([_message("Is a ZEBRA allowed?", intent="nlu_fallback")])
    assert msg.get("intent") == {"name": "intent_a", "confidence": 1.0}
    ranking = msg.get("intent_ranking")
    assert ranking[0] == {"name": "intent_a", "confidence": 1.0}
    assert [r["name"] for r in ranking].count("intent_a") == 1  # no duplicate entry
    assert msg.get("safety_gate") == "first"   # first matching category wins


def test_safety_gate_leaves_other_messages_to_the_model(tmp_path):
    [msg] = _gate(tmp_path).process([_message("hotels in Lisbon", intent="ask_eco_hotels")])
    assert msg.get("intent") == {"name": "ask_eco_hotels", "confidence": 0.9}
    assert msg.get("safety_gate") is None


def test_safety_gate_without_a_patterns_file_still_validates_input(tmp_path):
    from components.safety_gate import SafetyGate

    gate = SafetyGate({"patterns_file": str(tmp_path / "missing.yml"), "max_chars": 10})
    [ok, long_msg] = gate.process([_message("zebra"), _message("x" * 11)])
    assert ok.get("intent")["name"] == "greet"
    assert long_msg.get("intent")["name"] == "invalid_input"


def test_safety_patterns_file_points_at_domain_intents():
    import re
    import yaml

    path = REPO_ROOT / "data" / "safety_patterns.yml"
    if not path.exists():
        pytest.skip("data/safety_patterns.yml not written yet")
    domain_intents = set(yaml.safe_load((REPO_ROOT / "domain.yml").read_text())["intents"])
    for cat in yaml.safe_load(path.read_text())["categories"]:
        assert cat["intent"] in domain_intents, cat["name"]
        assert cat["patterns"], cat["name"]
        for pattern in cat["patterns"]:
            re.compile(pattern)


def test_every_new_safety_intent_has_a_rule_and_response():
    import yaml

    rules = yaml.safe_load((REPO_ROOT / "data" / "rules.yml").read_text())["rules"]
    responses = yaml.safe_load((REPO_ROOT / "domain.yml").read_text())["responses"]
    pairs = {}
    for rule in rules:
        steps = rule["steps"]
        if len(steps) == 2 and "intent" in steps[0] and "action" in steps[1]:
            pairs[steps[0]["intent"]] = steps[1]["action"]
    for intent in ("ask_booking", "ask_regulated_advice", "insult",
                   "ask_privacy", "invalid_input"):
        assert intent in pairs, intent
        assert pairs[intent] in responses, pairs[intent]
