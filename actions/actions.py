from typing import Any, Text, Dict, List, Optional, Tuple
import json
import os
import random
import time
import urllib.parse
import uuid
from pathlib import Path

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, EventType, ActiveLoop, UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction


# =============================================================================
# Phase 2 — external API config (Nominatim / Overpass / Wikipedia)
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from a local .env file into os.environ.

    Deliberately minimal — python-dotenv is not a project dependency, and
    docker-compose already injects the same file through `env_file:`. Values
    already present in the environment always win.
    """
    if not path.exists():
        return
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Volunteer-run APIs (Nominatim, Overpass) require an identifying User-Agent
# with a contact address. See Checks/README.md. A deployed instance should
# override ECO_USER_AGENT so complaints reach the operator, not the developer.
USER_AGENT = os.environ.get(
    "ECO_USER_AGENT",
    "EcoTravelAdvisor/1.0 (barislaca94@gmail.com)",
)
HTTP_HEADERS = {"User-Agent": USER_AGENT}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# In-memory cache for geocoded coordinates so we do not re-hit Nominatim for
# the same city on every user turn. Nominatim allows 1 req/sec — the sleep
# only applies when we actually call the API.
_GEOCODE_CACHE: Dict[str, Tuple[float, float]] = {}
_LAST_NOMINATIM_CALL: List[float] = [0.0]

# Pre-fetched Overpass POI data (see scripts/fetch_pois.py).
ECO_DATA_DIR = REPO_ROOT / "data" / "eco_data"


# Outcome of a geocoding attempt. "not_found" and "unavailable" must stay
# apart: telling a user that Berlin does not exist because Nominatim was rate
# limiting us is both wrong and, inside a form, a dead end — the slot gets
# rejected and the same question is asked forever.
GEOCODE_OK = "ok"
GEOCODE_NOT_FOUND = "not_found"
GEOCODE_UNAVAILABLE = "unavailable"


def geocode_city_result(city: str) -> Tuple[Optional[Tuple[float, float]], Text]:
    """Geocode a place name with Nominatim. Returns (coords, status).

    status is one of GEOCODE_OK / GEOCODE_NOT_FOUND / GEOCODE_UNAVAILABLE.
    Successful lookups are cached; the 1 request/second policy is honoured and
    the required User-Agent is sent. A rate-limit or maintenance answer is
    retried once, then reported as unavailable rather than as "no such place".
    """
    key = (city or "").strip().lower()
    if not key:
        return None, GEOCODE_NOT_FOUND
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key], GEOCODE_OK

    for attempt in (1, 2):
        # Rate limit — Nominatim policy is at most one request per second.
        delta = time.time() - _LAST_NOMINATIM_CALL[0]
        if delta < 1.0:
            time.sleep(1.0 - delta)

        try:
            response = requests.get(
                NOMINATIM_URL,
                params={"q": city, "format": "json", "limit": 1},
                headers=HTTP_HEADERS,
                timeout=10,
            )
        except requests.RequestException:
            _LAST_NOMINATIM_CALL[0] = time.time()
            if attempt == 1:
                time.sleep(1.5)
                continue
            return None, GEOCODE_UNAVAILABLE

        _LAST_NOMINATIM_CALL[0] = time.time()

        # 429 (rate limited) and 5xx bodies are HTML or plain text, so .json()
        # raises — which used to crash the whole action.
        if response.status_code != 200:
            if attempt == 1 and response.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5)
                continue
            return None, GEOCODE_UNAVAILABLE

        try:
            data = response.json()
        except ValueError:
            return None, GEOCODE_UNAVAILABLE

        if not data:
            return None, GEOCODE_NOT_FOUND
        try:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
        except (KeyError, ValueError, TypeError, IndexError):
            return None, GEOCODE_UNAVAILABLE

        _GEOCODE_CACHE[key] = (lat, lon)
        return (lat, lon), GEOCODE_OK

    return None, GEOCODE_UNAVAILABLE


def geocode_city(city: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for a city name, or None if it could not be resolved.

    Thin wrapper for callers that cannot act on the reason for a failure.
    """
    coords, _ = geocode_city_result(city)
    return coords


def load_eco_data(city: str, kind: str) -> List[Dict[str, Any]]:
    """Load pre-fetched Overpass POI data for a city.

    kind ∈ {"hotels", "transit", "attractions"}.
    Returns [] if the JSON file does not exist yet (pre-fetch not run for
    this city).
    """
    path = ECO_DATA_DIR / f"{city.lower()}_{kind}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


# =============================================================================
# Weather helpers + action
# =============================================================================

WEATHER_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
    3: "Overcast", 45: "Foggy", 61: "Light rain",
    63: "Moderate rain", 65: "Heavy rain", 71: "Light snow",
    80: "Rain showers", 95: "Thunderstorm",
}


def describe_wind(wind_speed: float) -> str:
    if wind_speed < 10:
        return "calm"
    if wind_speed < 30:
        return "light breeze"
    if wind_speed < 60:
        return "windy"
    return "very windy, be careful outdoors"


def describe_humidity(humidity: float) -> str:
    if humidity > 80:
        return "very damp"
    if humidity > 60:
        return "feels humid"
    return "comfortable"


def clothing_tip(temperature: float) -> str:
    if temperature < 5:
        return "wear a heavy coat"
    if temperature < 15:
        return "carry a light jacket"
    if temperature < 25:
        return "comfortable weather, dress normally"
    return "dress lightly and stay hydrated"


class ActionGetWeather(Action):

    def name(self) -> Text:
        return "action_get_weather"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        city = tracker.get_slot("city_name") or tracker.get_slot("destination")

        if not city:
            dispatcher.utter_message(text="Which city would you like the weather for?")
            return []

        coords, status = geocode_city_result(city)
        if not coords:
            if status == GEOCODE_UNAVAILABLE:
                dispatcher.utter_message(
                    text=(
                        "I couldn't reach the map service to look that place up. "
                        "Please try again in a moment."
                    )
                )
            else:
                dispatcher.utter_message(text=f"Sorry, I could not find a place called {city}.")
            return []
        latitude, longitude = coords

        try:
            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":  latitude,
                    "longitude": longitude,
                    "current":   "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
                    "daily":     "temperature_2m_max,precipitation_sum",
                    "timezone":  "auto",
                },
                timeout=10,
            )
            weather_data = weather_response.json()
        except (requests.RequestException, ValueError):
            dispatcher.utter_message(text="I could not reach the weather service. Please try again in a moment.")
            return []

        # Open-Meteo answers HTTP 400 with an {"error": true, "reason": ...}
        # body for bad coordinates, so a 200-shaped parse is not guaranteed.
        try:
            current       = weather_data["current"]
            temperature   = current["temperature_2m"]
            wind_speed    = current["wind_speed_10m"]
            humidity      = current["relative_humidity_2m"]
            weather_code  = current["weather_code"]
            tomorrow_max  = weather_data["daily"]["temperature_2m_max"][1]
            tomorrow_rain = weather_data["daily"]["precipitation_sum"][1]
        except (KeyError, IndexError, TypeError):
            dispatcher.utter_message(
                text=(
                    f"The weather service returned an unexpected answer for {city}. "
                    "Please try again in a moment."
                )
            )
            return []

        description = WEATHER_DESCRIPTIONS.get(weather_code, "Unknown conditions")
        wind_note = describe_wind(wind_speed)
        humidity_note = describe_humidity(humidity)
        tip = clothing_tip(temperature)

        if tomorrow_rain > 1:
            tomorrow_line = (
                f"Tomorrow: high of {tomorrow_max} C with {tomorrow_rain}mm rain — bring an umbrella."
            )
        else:
            tomorrow_line = f"Tomorrow: high of {tomorrow_max} C with no rain expected."

        reply = (
            f"It is {temperature} C and {description} in {city} right now.\n"
            f"Humidity: {humidity}% ({humidity_note}).\n"
            f"Wind: {wind_speed} km/h ({wind_note}).\n"
            f"Tip: {tip}.\n"
            f"{tomorrow_line}"
        )

        dispatcher.utter_message(text=reply)
        return []


# =============================================================================
# Currency action
# =============================================================================

class ActionGetCurrency(Action):

    def name(self) -> Text:
        return "action_get_currency"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        from_currency = tracker.get_slot("from_currency")
        to_currency = tracker.get_slot("to_currency")

        if not from_currency or not to_currency:
            dispatcher.utter_message(
                text="Which currencies should I convert? For example: 'convert USD to EUR'."
            )
            return []

        base = from_currency.upper()
        target = to_currency.upper()

        try:
            response = requests.get(
                "https://api.frankfurter.app/latest",
                params={"from": base, "to": target},
                timeout=10,
            )
            data = response.json()
        except (requests.RequestException, ValueError):
            dispatcher.utter_message(text="I could not reach the exchange rate service. Please try again in a moment.")
            return []

        rates = data.get("rates") if isinstance(data, dict) else None
        if not rates or target not in rates:
            dispatcher.utter_message(
                text=f"Sorry, I could not fetch an exchange rate from {base} to {target}."
            )
            return []

        rate = rates[target]
        date = data.get("date", "today")

        reply = f"1 {base} = {rate} {target} (as of {date})."
        dispatcher.utter_message(text=reply)
        return []


# =============================================================================
# Eco-travel: mock reference data (used instead of Amadeus/Climatiq keys)
# =============================================================================

# NOTE: The old ECO_HOTELS dict fabricated eco-certifications for named
# properties. That is exactly the greenwashing pattern the assignment brief
# forbids, so it has been removed. Hotel data now comes from real OSM records
# in `data/eco_data/<city>_hotels.json` (pre-fetched by scripts/fetch_pois.py).
# The bot presents PROXY sustainability signals (transit proximity, room count,
# parking availability) and explicitly disclaims that these are not
# certifications. See Checks/README.md § "The gap you will have to think about".


# A `public_transport=stop_position` node is often a single bus stop, which
# almost every city-centre hotel has within 50 m — so treating all transit
# alike made the signal meaningless (every hotel scored "~20m to transit").
# Rail, metro and tram access is the meaningful one for car-free travel.
RAIL_TAG_KEYS = ("subway", "train", "tram", "light_rail")


def _is_rail_stop(point: Dict[str, Any]) -> bool:
    """True if an OSM transit record is a rail/metro/tram stop, not a bus stop."""
    tags = point.get("tags") or {}
    if tags.get("railway") == "station":
        return True
    return any(tags.get(key) == "yes" for key in RAIL_TAG_KEYS)


def _nearest_metres(hotel: Dict[str, Any],
                    points: List[Dict[str, Any]]) -> Optional[int]:
    """Distance in metres to the closest of `points`, or None if unknown."""
    if not points:
        return None
    try:
        return int(min(
            _haversine_km(hotel["lat"], hotel["lon"], p["lat"], p["lon"]) * 1000
            for p in points
        ))
    except (KeyError, ValueError, TypeError):
        return None


def compute_proxy_signals(hotel: Dict[str, Any],
                          transit_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return proxy sustainability signals for a hotel from its OSM tags."""
    tags = hotel.get("tags", {})
    signals: Dict[str, Any] = {}

    # Not certifications — proxies for car-free access.
    rail_points = [p for p in transit_points if _is_rail_stop(p)]
    signals["nearest_rail_m"] = _nearest_metres(hotel, rail_points)
    signals["nearest_stop_m"] = _nearest_metres(hotel, transit_points)

    # Rooms count (small = smaller footprint per stay). Absent = unknown.
    rooms_raw = tags.get("rooms")
    try:
        signals["rooms"] = int(rooms_raw) if rooms_raw else None
    except (ValueError, TypeError):
        signals["rooms"] = None

    # Parking tag (car park = higher car use). Absent = unknown, not counted.
    # Measured coverage across the cached cities: under 2% of hotels carry it,
    # so it almost never contributes — kept because when present it is solid.
    signals["parking"] = tags.get("parking")

    # Stars — a price proxy, not an eco signal. Used only for budget fit.
    stars_raw = tags.get("stars")
    try:
        signals["stars"] = int(str(stars_raw)[0]) if stars_raw else None
    except (ValueError, TypeError, IndexError):
        signals["stars"] = None

    # Accessibility (42% coverage) — reported, never scored. The brief asks
    # for inclusivity, and step-free access is something a user can act on.
    signals["wheelchair"] = tags.get("wheelchair")

    return signals


STOP_WALK_M = 300    # a bus stop this close is still worth something


def signals_score(signals: Dict[str, Any]) -> Tuple[float, str]:
    """Combine the proxy signals into a 0-3 score with a colour band.

    Only counts positive evidence — absence of data does NOT lower the score,
    because OSM tags are missing far more often than they are false. Rail
    distance is graded rather than a yes/no test: in a dense city centre every
    hotel passes a single 800 m threshold, which ranks nothing.
    """
    positives = 0.0

    rail = signals.get("nearest_rail_m")
    stop = signals.get("nearest_stop_m")
    if rail is not None and rail < 300:
        positives += 1.0
    elif rail is not None and rail < 800:
        positives += 0.7
    elif rail is not None and rail < 1500:
        positives += 0.4
    elif stop is not None and stop < STOP_WALK_M:
        positives += 0.4

    rooms = signals.get("rooms")
    if rooms is not None and rooms < 30:
        positives += 1.0
    elif rooms is not None and rooms < 60:
        positives += 0.5

    if signals.get("parking") == "no":
        positives += 1.0

    if positives >= 1.5:
        return positives, "🟢"
    if positives >= 0.8:
        return positives, "🟡"
    return positives, "🔴"


def format_signals(signals: Dict[str, Any]) -> str:
    """Human-readable summary of the proxy signals we have for a hotel."""
    parts: List[str] = []
    rail = signals.get("nearest_rail_m")
    stop = signals.get("nearest_stop_m")
    if rail is not None:
        parts.append(f"~{rail}m to rail/metro")
    if stop is not None and (rail is None or stop < rail):
        parts.append(f"~{stop}m to a transit stop")
    if signals.get("rooms") is not None:
        parts.append(f"{signals['rooms']} rooms")
    if signals.get("parking"):
        parts.append(f"parking: {signals['parking']}")
    if signals.get("stars") is not None:
        parts.append(f"{signals['stars']}★")
    wheelchair = signals.get("wheelchair")
    if wheelchair:
        parts.append(f"step-free access: {wheelchair}")
    return " · ".join(parts) if parts else "no usable OSM tags recorded"


def rank_eco_hotels(city: str,
                    top_n: int = 5,
                    budget_per_night: Optional[float] = None) -> List[Dict[str, Any]]:
    """Return the top N hotels for a city ranked by proxy signals.

    `budget_per_night` (EUR) applies the only price signal OSM carries: a
    star rating. On a tight budget, 4-5★ properties are pushed down the list
    and the reason is reported back, rather than being silently dropped.
    Each returned dict has: name, band, score, signals, note.
    """
    hotels_file = load_eco_data(city, "hotels")
    if not hotels_file:
        return []
    records = hotels_file.get("records", []) if isinstance(hotels_file, dict) else []

    transit_file = load_eco_data(city, "transit")
    transit_records = transit_file.get("records", []) if isinstance(transit_file, dict) else []

    scored: List[Dict[str, Any]] = []
    for h in records:
        if not h.get("name"):
            continue
        signals = compute_proxy_signals(h, transit_records)
        score, band = signals_score(signals)

        note = None
        stars = signals.get("stars")
        if budget_per_night is not None and stars is not None and stars >= 4 \
                and budget_per_night < 120:
            score -= 1.0
            note = f"{stars}★ — likely above ~{budget_per_night:.0f} EUR/night"

        scored.append({
            "name":    h["name"],
            "score":   score,
            "band":    band,
            "signals": signals,
            "note":    note,
        })

    scored.sort(key=lambda r: (
        -r["score"],
        r["signals"].get("nearest_rail_m") or 99_000,
        r["signals"].get("nearest_stop_m") or 99_000,
    ))
    return scored[:top_n]


HOTEL_BAND_LABELS = {
    "🟢": "Strong proxy signals",
    "🟡": "Some proxy signals",
    "🔴": "Little evidence in OpenStreetMap",
}

# Nights implied by the trip_length slot, used to turn a total budget into a
# per-night figure for the star-rating price proxy.
TRIP_NIGHTS = {"weekend": 2, "week": 7, "extended": 14}

# Share of the total budget assumed to go on accommodation once travel,
# food and activities are accounted for. A rough planning heuristic, stated
# as such wherever it reaches the user.
ACCOMMODATION_BUDGET_SHARE = 0.5


def nightly_budget(tracker: Tracker) -> Optional[float]:
    """Approximate EUR available per night, or None when unknowable.

    Needs both a budget and a trip length; without the second, any per-night
    figure would be invented, so the price proxy is simply not applied.
    """
    budget_raw = tracker.get_slot("budget")
    nights = TRIP_NIGHTS.get(tracker.get_slot("trip_length") or "")
    if not budget_raw or not nights:
        return None
    digits = "".join(ch for ch in str(budget_raw) if ch.isdigit())
    if not digits:
        return None
    return int(digits) * ACCOMMODATION_BUDGET_SHARE / nights


def format_hotel_card(hotel: Dict[str, Any]) -> Text:
    """One colour-coded card per hotel, colour always paired with a word."""
    band = hotel["band"]
    lines = [
        f"{band} {hotel['name']} — {HOTEL_BAND_LABELS.get(band, '')}",
        format_signals(hotel["signals"]),
    ]
    if hotel.get("note"):
        lines.append(f"Budget note: {hotel['note']}")
    return "\n".join(lines)


SUPPORTED_CITIES_HINT = "Barcelona, Amsterdam, Kyoto, Lisbon, Berlin, Paris, Copenhagen, Oslo"

GREENWASHING_DISCLAIMER = (
    "⚠️ These are proxy signals from OpenStreetMap tags, not verified "
    "eco-certifications. Near-transit / small-scale / no-car-park correlate "
    "with a smaller footprint but do NOT guarantee a hotel is certified. "
    "If eco-certification is important to you, cross-check with an "
    "independent source (Green Key, EU Ecolabel, LEED)."
)


LOCAL_ACTIVITIES = {
    "barcelona": [
        "Guided walking tour of Gracia neighbourhood by local cooperative (small group, no bus)",
        "Cooking class with a Catalan family focusing on seasonal, local produce",
        "Bike tour along the beachfront run by a women-owned cooperative",
    ],
    "amsterdam": [
        "Canal clean-up boat tour (community volunteer scheme)",
        "Fair-trade coffee roastery visit and tasting",
        "Cycling tour of Amsterdam Noord's urban gardens and vintage markets",
    ],
    "kyoto": [
        "Tea ceremony hosted by a family-run machiya (traditional townhouse)",
        "Zen temple morning meditation with a local monk",
        "Bamboo forest guided walk with a community forester",
    ],
    "lisbon": [
        "Fado dinner supporting a local musicians' cooperative",
        "Portuguese tile painting workshop with a heritage artisan",
        "Guided tram walk exploring Alfama history with a local historian",
    ],
    "berlin": [
        "Urban gardening workshop at Prinzessinnengarten",
        "Alternative history tour by a local ex-Berliner (no vehicles)",
        "Cycling food tour featuring vegan and zero-waste eateries",
    ],
    "paris": [
        "Sustainable fashion walking tour of Le Marais",
        "Community bakery apprenticeship morning (make your own bread)",
        "Guided cycling tour of Canal Saint-Martin gardens",
    ],
    "copenhagen": [
        "Community-run kayak tour of the harbour",
        "Refill and zero-waste shopping tour in Norrebro",
        "Foraging walk in the city forest with a local ecologist",
    ],
    "oslo": [
        "Fjord kayaking with a local naturalist guide",
        "Sami cultural workshop supporting an indigenous cooperative",
        "Sustainable seafood cooking class with a Norwegian chef",
    ],
}


def nearby_cultural_sites(city: str, limit: int = 4) -> List[Dict[str, Any]]:
    """Museums, monuments and castles from the pre-fetched Overpass cache.

    The curated LOCAL_ACTIVITIES list is human-written and small; this adds
    the real OSM records for the same city (the brief's "cultural
    experiences"), closest to the city centre first.
    """
    data = load_eco_data(city, "attractions")
    records = data.get("records", []) if isinstance(data, dict) else []
    centre = data.get("centre") if isinstance(data, dict) else None
    if not records:
        return []

    def distance(record: Dict[str, Any]) -> float:
        if not centre:
            return 0.0
        try:
            return _haversine_km(
                centre["lat"], centre["lon"], record["lat"], record["lon"]
            )
        except (KeyError, TypeError, ValueError):
            return 99_999.0

    sites = []
    for record in sorted(records, key=distance):
        tags = record.get("tags") or {}
        kind = tags.get("tourism") or tags.get("historic") or ""
        if kind == "attraction":
            continue  # too generic to be useful on its own
        sites.append({"name": record["name"], "kind": kind.replace("_", " ")})
        if len(sites) >= limit:
            break
    return sites


CARBON_OFFSET_PROGRAMS = [
    {
        "name": "Gold Standard",
        "type": "Verified carbon credits (renewable energy, cookstoves, forestry)",
        "url": "https://www.goldstandard.org",
        "note": "One of the strictest offset standards, backed by WWF.",
    },
    {
        "name": "Atmosfair",
        "type": "Flight-focused offset via CDM-certified projects",
        "url": "https://www.atmosfair.de",
        "note": "German non-profit, strong on additionality reporting.",
    },
    {
        "name": "Klima",
        "type": "Consumer app for monthly carbon subscriptions",
        "url": "https://klima.com",
        "note": "Easy for individuals; portfolio of Gold Standard and VCS projects.",
    },
    {
        "name": "myclimate",
        "type": "Project-based offset (Swiss foundation)",
        "url": "https://www.myclimate.org",
        "note": "Transparent project catalogue, per-flight calculator.",
    },
]


# kg CO2e per passenger-km, used whenever Climatiq is unavailable.
#
# These were re-derived on 2026-09-16 from the same BEIS/UBA factors Climatiq
# serves (measured with scripts/climatiq_check.py) so that a fallback answer
# does not contradict a live one. The earlier table used a domestic-flight
# figure (0.255) for every flight, which roughly doubled short-haul estimates.
# Ferry, bicycle and walk are not covered by the activity ids above and keep
# their Our World in Data values.
EMISSION_FACTORS = {
    "flight":  0.126,
    "plane":   0.126,
    "car":     0.164,
    "bus":     0.040,
    "train":   0.031,
    "coach":   0.040,
    "ferry":   0.019,
    "bicycle": 0.0,
    "walk":    0.0,
}

LOCAL_SOURCE_LABEL = "local BEIS 2023 / Our World in Data table"


# -----------------------------------------------------------------------------
# Climatiq — live carbon estimates, with the local table as fallback
# -----------------------------------------------------------------------------
# The brief asks for "the Climatiq API for real-time carbon emission
# calculations per transport mode". Climatiq needs a (free) key, so the bot
# must work without one: every call degrades to EMISSION_FACTORS above and
# SAYS which source produced the number, so a user is never shown a live-API
# figure that was actually a local estimate.

CLIMATIQ_BATCH_URL = "https://api.climatiq.io/data/v1/estimate/batch"
CLIMATIQ_SEARCH_URL = "https://api.climatiq.io/data/v1/search"
CLIMATIQ_DATA_VERSION = os.environ.get("CLIMATIQ_DATA_VERSION", "^21")

# Keep the request well inside the brief's "under three seconds for critical
# interactions" budget — one batched call, short timeout, then fall back.
CLIMATIQ_TIMEOUT = 2.5

# Activity IDs are resolved with `python scripts/climatiq_check.py`, which
# queries Climatiq's /search endpoint and verifies each id still estimates.
# Verified against data version ^21 on 2026-09-16 with scripts/climatiq_check.py.
# "bus" maps to the COACH factor on purpose: the bot compares city-to-city
# options, and Climatiq's local_bus factor (0.10 kg/pkm) models urban
# stop-start driving, not intercity travel.
CLIMATIQ_ACTIVITY_IDS: Dict[str, str] = {
    "train":  "passenger_train-route_type_national_rail-fuel_source_na",
    "bus":    "passenger_vehicle-vehicle_type_coach-fuel_source_na-distance_na-engine_size_na",
    "coach":  "passenger_vehicle-vehicle_type_coach-fuel_source_na-distance_na-engine_size_na",
    "car":    "passenger_vehicle-vehicle_type_car-fuel_source_na-engine_size_na-vehicle_age_na-vehicle_weight_na",
}

# Aviation is the one mode where a single average is badly misleading: BEIS
# publishes separate short- and long-haul factors, and the undifferentiated
# "distance_na" factor (0.109 kg/pkm) understates a short hop by roughly a
# third. Both include radiative forcing and the distance uplift.
CLIMATIQ_SHORT_HAUL_KM = 3700
CLIMATIQ_FLIGHT_IDS = {
    "short": (
        "passenger_flight-route_type_international-aircraft_type_na"
        "-distance_short_haul_lt_3700km-class_economy-rf_included-distance_uplift_included"
    ),
    "long": (
        "passenger_flight-route_type_international-aircraft_type_na"
        "-distance_long_haul_gt_3700km-class_economy-rf_included-distance_uplift_included"
    ),
}


def climatiq_activity_id(mode: Text, distance_km: float) -> Optional[Text]:
    """Return the Climatiq activity id for a mode, or None if uncovered."""
    m = (mode or "").lower()
    if m in ("flight", "plane"):
        band = "short" if distance_km < CLIMATIQ_SHORT_HAUL_KM else "long"
        return CLIMATIQ_FLIGHT_IDS[band]
    return CLIMATIQ_ACTIVITY_IDS.get(m)

# (mode, rounded km) -> kg CO2e, so a follow-up question in the same
# conversation does not spend another API call.
_CARBON_CACHE: Dict[Tuple[str, int], float] = {}


def climatiq_api_key() -> Optional[str]:
    """Return the Climatiq key from the environment, or None if unset."""
    key = os.environ.get("CLIMATIQ_API_KEY", "").strip()
    return key or None


def local_carbon(mode: Text, distance_km: float) -> float:
    """Carbon estimate from the built-in DEFRA factor table."""
    factor = EMISSION_FACTORS.get(mode.lower(), EMISSION_FACTORS["flight"])
    return distance_km * factor


def estimate_carbon(modes: List[Text],
                    distance_km: float) -> Tuple[Dict[Text, float], Text]:
    """Return {mode: kg CO2e} plus a label naming the source of the numbers.

    Uses one batched Climatiq call when CLIMATIQ_API_KEY is set; on a missing
    key, HTTP error, timeout or unparseable answer it falls back to
    EMISSION_FACTORS. Modes Climatiq does not cover (walk, bicycle, ferry)
    always come from the local table.
    """
    rounded = int(round(distance_km))
    results: Dict[Text, float] = {}
    from_live: set = set()

    key = climatiq_api_key()
    wanted = [m for m in modes if climatiq_activity_id(m, distance_km)]

    # Serve whatever we already looked up in this process.
    if key:
        for mode in list(wanted):
            cached = _CARBON_CACHE.get((mode.lower(), rounded))
            if cached is not None:
                results[mode] = cached
                from_live.add(mode)
                wanted.remove(mode)

    if key and wanted:
        payload = [
            {
                "emission_factor": {
                    "activity_id": climatiq_activity_id(mode, distance_km),
                    "data_version": CLIMATIQ_DATA_VERSION,
                },
                "parameters": {
                    "passengers": 1,
                    "distance": round(distance_km, 2),
                    "distance_unit": "km",
                },
            }
            for mode in wanted
        ]
        try:
            response = requests.post(
                CLIMATIQ_BATCH_URL,
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
                timeout=CLIMATIQ_TIMEOUT,
            )
            body = response.json()
        except (requests.RequestException, ValueError):
            body = None

        # The batch endpoint answers {"results": [...]}; some builds answer
        # with a bare list. Accept either, and treat per-item errors as misses.
        items: Optional[List[Any]] = None
        if isinstance(body, dict) and isinstance(body.get("results"), list):
            items = body["results"]
        elif isinstance(body, list):
            items = body

        for mode, item in zip(wanted, items or []):
            co2e = item.get("co2e") if isinstance(item, dict) else None
            if co2e is None:
                continue
            unit = item.get("co2e_unit", "kg")
            try:
                kg = float(co2e) * (1000.0 if unit == "t" else 1.0)
            except (TypeError, ValueError):
                continue
            _CARBON_CACHE[(mode.lower(), rounded)] = kg
            results[mode] = kg
            from_live.add(mode)

    # Anything still missing (no key, API problem, or a mode Climatiq does not
    # cover) comes from the local table.
    for mode in modes:
        if mode not in results:
            results[mode] = local_carbon(mode, distance_km)

    climatiq_capable = {m for m in modes if climatiq_activity_id(m, distance_km)}
    if not key:
        label = f"Source: {LOCAL_SOURCE_LABEL} (no Climatiq key configured)."
    elif from_live >= climatiq_capable and climatiq_capable:
        label = f"Source: Climatiq live estimate, data version {CLIMATIQ_DATA_VERSION}."
    elif from_live:
        label = (
            f"Source: Climatiq (data version {CLIMATIQ_DATA_VERSION}) where available, "
            f"otherwise {LOCAL_SOURCE_LABEL}."
        )
    else:
        label = f"Source: {LOCAL_SOURCE_LABEL} (Climatiq unavailable just now)."
    return results, label


def eco_score_to_band(score: float) -> str:
    """Return a colour-coded emoji band based on eco score (0-100)."""
    if score >= 85:
        return "🟢"
    if score >= 60:
        return "🟡"
    return "🔴"


def carbon_to_band(kg_co2: float) -> str:
    """Colour code for the absolute size of a trip's footprint (kg CO2e)."""
    if kg_co2 < 50:
        return "🟢"
    if kg_co2 < 250:
        return "🟡"
    return "🔴"


# kg CO2e per passenger-km. Rail and coach sit around 0.03-0.04, car and
# aviation around 0.13-0.16, so these thresholds separate the modes rather
# than the trip lengths.
INTENSITY_GREEN = 0.05
INTENSITY_AMBER = 0.10


def intensity_band(kg_co2: float, distance_km: float) -> str:
    """Colour code for how carbon-intensive a mode is, per passenger-km.

    Comparing modes on absolute kg would mark every long journey red and
    every short one green, which says more about the distance than about the
    choice the user is actually making.
    """
    if distance_km <= 0:
        return carbon_to_band(kg_co2)
    intensity = kg_co2 / distance_km
    if intensity < INTENSITY_GREEN:
        return "🟢"
    if intensity < INTENSITY_AMBER:
        return "🟡"
    return "🔴"


def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Great-circle distance between two points in kilometres."""
    from math import radians, sin, cos, asin, sqrt

    lat1, lon1 = radians(a_lat), radians(a_lon)
    lat2, lon2 = radians(b_lat), radians(b_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def approx_distance_km(origin: str, destination: str) -> float:
    """Rough distance between two named places, or 0.0 if either is unknown."""
    km, _ = approx_distance_result(origin, destination)
    return km


def approx_distance_result(origin: str, destination: str) -> Tuple[float, Text]:
    """Distance in km plus the geocoding status, so callers can explain a zero.

    Status is GEOCODE_OK, GEOCODE_NOT_FOUND (a place we genuinely cannot find)
    or GEOCODE_UNAVAILABLE (the map service is down or rate limiting us).
    """
    a, status_a = geocode_city_result(origin)
    b, status_b = geocode_city_result(destination)
    if a and b:
        return _haversine_km(a[0], a[1], b[0], b[1]), GEOCODE_OK
    if GEOCODE_UNAVAILABLE in (status_a, status_b):
        return 0.0, GEOCODE_UNAVAILABLE
    return 0.0, GEOCODE_NOT_FOUND


MAP_UNAVAILABLE_MESSAGE = (
    "I couldn't reach the map service just now, so I can't work out the "
    "distance. Please try again in a moment."
)


# =============================================================================
# Weighted ranking of travel options (carbon + cost + stated preferences)
# =============================================================================
# The brief asks for "a weighted scoring function that combines carbon impact,
# price, and user-stated preferences". Carbon comes from Climatiq (or the local
# table). Price is the weak link: Amadeus was decommissioned and no free API
# gives per-route fares, so the bot uses published average ticket costs and
# labels every figure as an estimate rather than implying a live quote.

# EUR per passenger-km plus a fixed component (booking fees, airport transfer,
# station access). Indicative European averages, 2026.
EST_COST_EUR_PER_KM = {
    "train":  0.14,
    "bus":    0.06,
    "coach":  0.06,
    "car":    0.12,   # fuel + wear, single occupant
    "flight": 0.09,
    "ferry":  0.10,
}
EST_COST_FIXED_EUR = {
    "train":  5.0,
    "bus":    2.0,
    "coach":  2.0,
    "car":    0.0,
    "flight": 45.0,   # airport transfer and fees dominate short hops
    "ferry":  10.0,
}

# (carbon weight, cost weight) per stated sustainability level. A user who
# says sustainability matters most gets carbon weighted four times as heavily
# as price; a budget-first user gets the reverse.
SUSTAINABILITY_WEIGHTS = {
    "high":   (0.8, 0.2),
    "medium": (0.5, 0.5),
    "low":    (0.3, 0.7),
}

DEFAULT_MODES = ("train", "bus", "car", "flight")


def estimate_cost_eur(mode: Text, distance_km: float) -> float:
    """Indicative one-way ticket cost. NOT a live fare — always label it."""
    m = mode.lower()
    per_km = EST_COST_EUR_PER_KM.get(m, 0.10)
    return EST_COST_FIXED_EUR.get(m, 0.0) + per_km * distance_km


def _filter_modes(modes: List[Text],
                  transport_preference: Optional[Text]) -> Tuple[List[Text], List[Text]]:
    """Apply the user's stated transport preference.

    Returns (kept, excluded) so the caller can tell the user what was dropped
    and why, instead of silently hiding options.
    """
    if transport_preference == "train_or_bus":
        kept = [m for m in modes if m.lower() in ("train", "bus", "coach")]
    elif transport_preference == "any_low_carbon":
        kept = [m for m in modes if m.lower() not in ("flight", "plane")]
    else:
        return list(modes), []
    excluded = [m for m in modes if m not in kept]
    return (kept or list(modes)), (excluded if kept else [])


def score_transport_options(
    distance_km: float,
    sustainability_level: Optional[Text] = None,
    transport_preference: Optional[Text] = None,
    budget: Optional[Text] = None,
    modes: Tuple[Text, ...] = DEFAULT_MODES,
) -> Tuple[List[Dict[Text, Any]], List[Text], Text]:
    """Rank travel modes by a weighted carbon/cost score.

    Returns (ranked options, excluded modes, carbon source label). Each option
    carries mode, carbon_kg, band, cost_eur, score and any budget warning.
    """
    kept, excluded = _filter_modes(list(modes), transport_preference)
    carbon, source_label = estimate_carbon(kept, distance_km)

    costs = {m: estimate_cost_eur(m, distance_km) for m in kept}
    c_values = [carbon[m] for m in kept]
    p_values = [costs[m] for m in kept]
    c_lo, c_hi = min(c_values), max(c_values)
    p_lo, p_hi = min(p_values), max(p_values)

    w_carbon, w_cost = SUSTAINABILITY_WEIGHTS.get(
        (sustainability_level or "medium").lower(), SUSTAINABILITY_WEIGHTS["medium"]
    )

    budget_eur = None
    if budget:
        digits = "".join(ch for ch in str(budget) if ch.isdigit())
        budget_eur = int(digits) if digits else None

    options: List[Dict[Text, Any]] = []
    for mode in kept:
        # Min-max normalise within this comparison, then invert so that 1.0 is
        # always "best on this axis". A single candidate scores 1.0 by default.
        c_norm = 1.0 if c_hi == c_lo else (carbon[mode] - c_lo) / (c_hi - c_lo)
        p_norm = 1.0 if p_hi == p_lo else (costs[mode] - p_lo) / (p_hi - p_lo)
        score = w_carbon * (1 - c_norm) + w_cost * (1 - p_norm)

        warning = None
        if budget_eur and costs[mode] * 2 > budget_eur * 0.5:
            # Return trip vs half the total budget.
            warning = (
                f"return travel ~{costs[mode] * 2:.0f} EUR — over half your "
                f"{budget_eur} EUR budget"
            )

        options.append({
            "mode":      mode,
            "carbon_kg": carbon[mode],
            "band":      intensity_band(carbon[mode], distance_km),
            "cost_eur":  costs[mode],
            "score":     score,
            "warning":   warning,
        })

    options.sort(key=lambda o: -o["score"])
    return options, excluded, source_label


BAND_LABELS = {
    "🟢": "Low emission",
    "🟡": "Moderate emission",
    "🔴": "High emission",
}


def format_transport_option(option: Dict[Text, Any], recommended: bool = False) -> Text:
    """One colour-coded card per travel option.

    The band emoji is always paired with a word, so the colour is never the
    only carrier of meaning (accessibility / screen readers).
    """
    band = option["band"]
    head = f"{band} {option['mode'].title()} — {BAND_LABELS.get(band, '')}"
    if recommended:
        head += "  ·  RECOMMENDED"
    lines = [
        head,
        f"~{option['carbon_kg']:.1f} kg CO2e per passenger · "
        f"~{option['cost_eur']:.0f} EUR one way (estimate) · "
        f"match score {option['score']:.2f}",
    ]
    if option["warning"]:
        lines.append(f"⚠️ {option['warning']}")
    return "\n".join(lines)


COST_DISCLAIMER = (
    "Costs are indicative averages per kilometre, not live fares — no free "
    "fare API is available since Amadeus was decommissioned. Treat them as an "
    "order of magnitude and check a booking site before deciding."
)


# =============================================================================
# Eco-travel actions
# =============================================================================

class ActionSuggestEcoHotels(Action):
    """List hotels in the requested city, ranked by proxy sustainability
    signals from OpenStreetMap. Never claims a hotel is 'eco-certified' —
    OSM has no reliable certification tags (see Checks/README.md).
    """

    def name(self) -> Text:
        return "action_suggest_eco_hotels"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        destination = tracker.get_slot("destination") or tracker.get_slot("city_name")

        if not destination:
            dispatcher.utter_message(text="Which city are you looking for hotels in?")
            return []

        top = rank_eco_hotels(
            destination,
            top_n=5,
            budget_per_night=nightly_budget(tracker),
        )
        if not top:
            dispatcher.utter_message(
                text=(
                    f"I don't have pre-fetched OSM data for {destination} yet. "
                    f"Supported cities: {SUPPORTED_CITIES_HINT}. "
                    "Ask an admin to run `python scripts/fetch_pois.py` to add it."
                )
            )
            return []

        dispatcher.utter_message(
            text=(
                f"Hotels in {destination.title()}, ranked by proxy sustainability "
                "signals — rail/metro proximity, small scale, no car park. "
                "Star ratings and step-free access are shown where OSM records them."
            )
        )
        # One message per hotel so each renders as its own colour-coded card.
        for hotel in top:
            dispatcher.utter_message(text=format_hotel_card(hotel))

        buttons = [
            {"title": f"Community activities in {destination.title()}",
             "payload": f'/ask_local_activities{{"destination":"{destination}"}}'},
            {"title": f"About {destination.title()}",
             "payload": f'/ask_place_description{{"destination":"{destination}"}}'},
            {"title": "Carbon offset options",
             "payload": "/ask_carbon_offset"},
        ]
        dispatcher.utter_message(text=GREENWASHING_DISCLAIMER, buttons=buttons)
        return [SlotSet("destination", destination)]


class ActionDescribePlace(Action):
    """Fetch a short Wikipedia summary for a destination."""

    def name(self) -> Text:
        return "action_describe_place"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        destination = tracker.get_slot("destination") or tracker.get_slot("city_name")
        if not destination:
            dispatcher.utter_message(text="Which place should I describe?")
            return []

        title = urllib.parse.quote(destination.strip().replace(" ", "_"))
        try:
            response = requests.get(
                WIKIPEDIA_SUMMARY_URL + title,
                headers=HTTP_HEADERS,
                timeout=8,
            )
            data = response.json()
        except (requests.RequestException, ValueError):
            dispatcher.utter_message(
                text="I could not reach Wikipedia right now. Please try again in a moment."
            )
            return []

        if response.status_code == 404 or data.get("type") == "disambiguation":
            dispatcher.utter_message(
                text=f"Sorry, Wikipedia does not have a clear summary page for {destination}."
            )
            return []

        extract = data.get("extract")
        page_url = ((data.get("content_urls") or {}).get("desktop") or {}).get("page")
        if not extract:
            dispatcher.utter_message(text=f"Wikipedia returned no summary for {destination}.")
            return []

        lines = [f"About {data.get('title', destination.title())}:", "", extract]
        if page_url:
            lines.append("")
            lines.append(f"Source: {page_url}")

        buttons: List[Dict[str, str]] = []
        if load_eco_data(destination, "hotels"):
            buttons.append({
                "title": f"Hotels in {destination.title()}",
                "payload": f'/ask_eco_hotels{{"destination":"{destination}"}}',
            })
        buttons.append({
            "title": f"Community activities in {destination.title()}",
            "payload": f'/ask_local_activities{{"destination":"{destination}"}}',
        })

        dispatcher.utter_message(text="\n".join(lines), buttons=buttons or None)
        return [SlotSet("destination", destination)]


class ActionSuggestTransport(Action):

    def name(self) -> Text:
        return "action_suggest_transport"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        origin = tracker.get_slot("origin")
        destination = tracker.get_slot("destination") or tracker.get_slot("city_name")

        if not origin or not destination:
            dispatcher.utter_message(
                text="Please tell me both the origin and destination. Example: 'green transport from London to Paris'."
            )
            return []

        distance, geo_status = approx_distance_result(origin, destination)
        if distance == 0.0:
            dispatcher.utter_message(
                text=MAP_UNAVAILABLE_MESSAGE if geo_status == GEOCODE_UNAVAILABLE
                else f"Sorry, I couldn't locate {origin} or {destination} to estimate distance."
            )
            return []

        # Preferences carry over from the trip planning form when the user has
        # filled it in this conversation; otherwise the weights default to
        # "medium" and nothing is filtered out.
        options, excluded, source_label = score_transport_options(
            distance,
            sustainability_level=tracker.get_slot("sustainability_level"),
            transport_preference=tracker.get_slot("transport_preference"),
            budget=tracker.get_slot("budget"),
        )

        header = [f"{origin.title()} → {destination.title()}: about {distance:.0f} km."]
        level = tracker.get_slot("sustainability_level")
        if level:
            w_c, w_p = SUSTAINABILITY_WEIGHTS.get(level, SUSTAINABILITY_WEIGHTS["medium"])
            header.append(
                f"Ranked for '{level}' sustainability — carbon weighted "
                f"{w_c:.0%}, cost {w_p:.0%}."
            )
        else:
            header.append("Ranked by an even split of carbon impact and cost.")
        if excluded:
            header.append(
                f"Left out at your request: {', '.join(m.title() for m in excluded)}."
            )
        dispatcher.utter_message(text=" ".join(header))

        # One message per option so the UI can colour each as its own card.
        for index, option in enumerate(options):
            dispatcher.utter_message(
                text=format_transport_option(option, recommended=(index == 0))
            )

        # Alert on the high-emission option, quantified against the best one.
        worst = max(options, key=lambda o: o["carbon_kg"])
        best = min(options, key=lambda o: o["carbon_kg"])
        if worst["band"] == "🔴" and worst is not best:
            factor = worst["carbon_kg"] / best["carbon_kg"] if best["carbon_kg"] else 0
            dispatcher.utter_message(
                text=(
                    f"⚠️ {worst['mode'].title()} emits about "
                    f"{worst['carbon_kg'] - best['carbon_kg']:.0f} kg CO2e more than "
                    f"{best['mode']}"
                    + (f" — roughly {factor:.0f}x higher." if factor else ".")
                )
            )

        buttons = [
            {"title": "Carbon offset options", "payload": "/ask_carbon_offset"},
            {"title": f"Eco hotels in {destination.title()}",
             "payload": f'/ask_eco_hotels{{"destination":"{destination}"}}'},
        ]
        dispatcher.utter_message(
            text=f"ℹ️ {source_label} {COST_DISCLAIMER}",
            buttons=buttons,
        )
        return []


class ActionCalculateCarbon(Action):

    def name(self) -> Text:
        return "action_calculate_carbon"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        origin = tracker.get_slot("origin")
        destination = tracker.get_slot("destination") or tracker.get_slot("city_name")
        transport_mode = tracker.get_slot("transport_mode") or "flight"

        if not origin or not destination:
            dispatcher.utter_message(
                text="I need both an origin and a destination. Example: 'carbon footprint of a flight from London to Paris'."
            )
            return []

        distance, geo_status = approx_distance_result(origin, destination)
        if distance == 0.0:
            dispatcher.utter_message(
                text=MAP_UNAVAILABLE_MESSAGE if geo_status == GEOCODE_UNAVAILABLE
                else f"Sorry, I couldn't locate {origin} or {destination}."
            )
            return []

        mode = transport_mode.lower()
        carbon, source_label = estimate_carbon([mode], distance)
        kg = carbon[mode]
        band = intensity_band(kg, distance)
        reply = (
            f"{band} {transport_mode.title()} from {origin.title()} to "
            f"{destination.title()} ({distance:.0f} km): about {kg:.1f} kg CO2e "
            f"per passenger — {BAND_LABELS.get(band, '')}.\n"
            f"That is {kg / distance:.3f} kg per passenger-km; the colour bands "
            f"are under {INTENSITY_GREEN} green, under {INTENSITY_AMBER} amber, "
            f"above that red.\n"
            f"{source_label}"
        )
        dispatcher.utter_message(text=reply)
        return []


class ActionSuggestActivities(Action):

    def name(self) -> Text:
        return "action_suggest_activities"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        destination = tracker.get_slot("destination") or tracker.get_slot("city_name")

        if not destination:
            dispatcher.utter_message(text="Which city are you interested in?")
            return []

        acts = LOCAL_ACTIVITIES.get(destination.lower())
        sites = nearby_cultural_sites(destination)

        if not acts and not sites:
            dispatcher.utter_message(
                text=(
                    f"I don't have community-supported activity ideas for {destination} yet. "
                    "Try Barcelona, Amsterdam, Kyoto, Lisbon, Berlin, Paris, Copenhagen, or Oslo."
                )
            )
            return []

        if acts:
            lines = [f"🟢 Community-friendly, low-impact activities in {destination.title()}:"]
            lines.extend(f"• {a}" for a in acts)
            lines.append("These options prioritise local businesses and avoid mass tourism.")
            dispatcher.utter_message(text="\n".join(lines))

        if sites:
            lines = [
                f"Cultural sites near the centre of {destination.title()}, "
                "from OpenStreetMap:"
            ]
            for site in sites:
                kind = site["kind"]
                lines.append(f"• {site['name']}" + (f" ({kind})" if kind else ""))
            lines.append(
                "Listed because OSM records them as museums, monuments or castles — "
                "no ranking or endorsement implied."
            )
            dispatcher.utter_message(text="\n".join(lines))

        return []


class ActionCarbonOffsetPrograms(Action):

    def name(self) -> Text:
        return "action_carbon_offset_programs"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        lines = ["Verified carbon offset programs:"]
        for p in CARBON_OFFSET_PROGRAMS:
            lines.append(f"🟢 {p['name']} — {p['type']} ({p['url']}). {p['note']}")
        lines.append(
            "Ethical caveat: offsets should complement, not replace, choosing low-emission travel in the first place."
        )
        dispatcher.utter_message(text="\n".join(lines))
        return []


def _build_transcript(tracker: Tracker, limit: int = 20) -> List[str]:
    """Build a chronological transcript from tracker events (last N turns)."""
    transcript: List[str] = []
    for event in tracker.events:
        event_type = event.get("event")
        text = event.get("text")
        if not text:
            continue
        if event_type == "user":
            transcript.append(f"USER: {text}")
        elif event_type == "bot":
            transcript.append(f"BOT : {text}")
    return transcript[-limit:]


def _build_handover_package(tracker: Tracker, reason: Text = "user_requested") -> Dict[Text, Any]:
    """Package the full conversation context for a human advisor handover.

    Follows the assignment brief: "Packaging full conversation context for
    human advisor handover" (Making a Bot Behave §5.3).
    """
    latest_intent = (tracker.latest_message or {}).get("intent", {}) or {}
    transcript = _build_transcript(tracker, limit=20)
    filled_slots = {
        name: value
        for name, value in (tracker.current_slot_values() or {}).items()
        if value not in (None, "", False)
    }
    turn_count = sum(1 for e in tracker.events if e.get("event") == "user")

    # The trip form clears its own slots on submit, so a handover straight
    # after a completed plan would otherwise reach the advisor with nothing
    # but a destination. last_trip_summary carries the plan itself.
    last_trip = filled_slots.pop("last_trip_summary", None)
    if isinstance(last_trip, str):
        try:
            last_trip = json.loads(last_trip)
        except json.JSONDecodeError:
            pass

    return {
        "ticket_id":         f"TR-{uuid.uuid4().hex[:6].upper()}",
        "reason":            reason,
        "conversation_id":   tracker.sender_id,
        "last_intent":       latest_intent.get("name"),
        "last_confidence":   latest_intent.get("confidence"),
        "turn_count":        turn_count,
        "collected_slots":   filled_slots,
        "last_trip_plan":    last_trip,
        "transcript":        transcript,
    }


class ActionHumanHandover(Action):
    """Escalate to a human travel advisor with full conversation context.

    The `package` dict below is printed to the actions-server console. A real
    system would forward it to Slack, Zendesk, or an email queue — see the
    project README for the integration stub.
    """

    def name(self) -> Text:
        return "action_human_handover"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        advisor = random.choice(["Anna", "Marco", "Priya", "Ling", "Ines"])
        package = _build_handover_package(tracker, reason="user_requested")

        # Print the full package for the advisor pipeline (demo stub).
        print("=" * 60, flush=True)
        print("HUMAN HANDOVER PACKAGE (user_requested)", flush=True)
        print(json.dumps(package, indent=2, default=str), flush=True)
        print("=" * 60, flush=True)

        lines = [
            f"🎫 HANDOVER · Ticket {package['ticket_id']} — connecting you to travel advisor {advisor}.",
            "Advisor context bundle:",
        ]
        for k, v in package["collected_slots"].items():
            lines.append(f"  • {k.replace('_', ' ').title()}: {v}")
        trip = package.get("last_trip_plan")
        if isinstance(trip, dict):
            lines.append(
                "  • Last trip plan: "
                + ", ".join(
                    f"{k.replace('_', ' ')}={v}"
                    for k, v in trip.items()
                    if v not in (None, "", [])
                )
            )
        if package["last_intent"]:
            conf = package["last_confidence"]
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
            lines.append(f"  • Last intent: {package['last_intent']} (confidence {conf_str})")
        lines.append(f"  • Turn count: {package['turn_count']}")
        lines.append("Response time: usually under 15 minutes during business hours.")

        dispatcher.utter_message(text="\n".join(lines))
        return [SlotSet("handover_active", True)]


class ActionSubmitTripForm(Action):

    def name(self) -> Text:
        return "action_submit_trip_form"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        destination = tracker.get_slot("destination") or "your destination"
        origin = tracker.get_slot("origin")
        dates = tracker.get_slot("travel_dates") or "your dates"
        budget = tracker.get_slot("budget")
        level = tracker.get_slot("sustainability_level") or "medium"
        preference = tracker.get_slot("transport_preference")
        trip_length = tracker.get_slot("trip_length")
        nights = TRIP_NIGHTS.get(trip_length or "")

        # ---- Travel options: every answer the form collected feeds this ----
        travel_options: List[Dict[Text, Any]] = []
        excluded: List[Text] = []
        carbon_source = ""
        distance = approx_distance_km(origin, destination) if origin else 0.0
        if distance:
            travel_options, excluded, carbon_source = score_transport_options(
                distance,
                sustainability_level=level,
                transport_preference=preference,
                budget=budget,
            )

        best = travel_options[0] if travel_options else None
        header_band = best["band"] if best else "🟢"

        summary = [
            f"{header_band} Trip plan — {destination.title()}"
            + (f" from {origin.title()}" if origin else ""),
            f"Dates: {dates}"
            + (f" · {trip_length} ({nights} nights)" if nights else "")
            + (f" · budget ~{budget} EUR" if budget else ""),
            f"Sustainability priority: {level}"
            + (f" · transport preference: {preference.replace('_', ' ')}" if preference else ""),
        ]
        if best:
            summary.append(
                f"Recommended way to get there: {best['mode']} "
                f"(~{best['carbon_kg']:.0f} kg CO2e, ~{best['cost_eur']:.0f} EUR one way)."
            )
        dispatcher.utter_message(text="\n".join(summary))

        # ---- Transport cards ----
        if travel_options:
            for index, option in enumerate(travel_options[:3]):
                dispatcher.utter_message(
                    text=format_transport_option(option, recommended=(index == 0))
                )
            if excluded:
                dispatcher.utter_message(
                    text=(
                        f"ℹ️ {', '.join(m.title() for m in excluded)} left out because "
                        f"you asked for {preference.replace('_', ' ')}. {carbon_source} "
                        f"{COST_DISCLAIMER}"
                    )
                )
            else:
                dispatcher.utter_message(text=f"ℹ️ {carbon_source} {COST_DISCLAIMER}")
        elif not origin:
            dispatcher.utter_message(
                text="ℹ️ Tell me where you are travelling from and I can compare "
                     "transport options and their carbon cost."
            )

        # ---- Hotels: more options when sustainability is the priority ----
        per_night = nightly_budget(tracker)
        ranked = rank_eco_hotels(
            destination,
            top_n=3 if level == "high" else 2,
            budget_per_night=per_night,
        )
        if ranked:
            head = "Where to stay — ranked by proxy sustainability signals:"
            if per_night:
                head += (
                    f" assuming roughly {per_night:.0f} EUR a night "
                    f"({ACCOMMODATION_BUDGET_SHARE:.0%} of your budget over {nights} nights)."
                )
            dispatcher.utter_message(text=head)
            for hotel in ranked:
                dispatcher.utter_message(text=format_hotel_card(hotel))
            if per_night and per_night < 60:
                dispatcher.utter_message(
                    text=(
                        f"⚠️ About {per_night:.0f} EUR a night is tight for a hotel in "
                        f"{destination.title()} — guesthouses and hostels are worth a look, "
                        "and they tend to be smaller-scale anyway."
                    )
                )
            dispatcher.utter_message(text=GREENWASHING_DISCLAIMER)

        # ---- Activities and offsetting ----
        acts = LOCAL_ACTIVITIES.get(destination.lower(), [])
        if acts:
            lines = ["Community-friendly activities:"]
            lines.extend(f"• {a}" for a in acts[:3])
            dispatcher.utter_message(text="\n".join(lines))

        top_offset = CARBON_OFFSET_PROGRAMS[0]
        offset_line = f"Suggested offset: {top_offset['name']} ({top_offset['url']})."
        if best:
            offset_line += (
                " Offsetting is a last step, though — the mode you pick matters more."
            )

        buttons = [
            {"title": f"Green transport to {destination.title()}",
             "payload": f'/ask_green_transport{{"destination":"{destination}"}}'},
            {"title": f"Community activities in {destination.title()}",
             "payload": f'/ask_local_activities{{"destination":"{destination}"}}'},
            {"title": "Talk to a human advisor",
             "payload": "/request_human_advisor"},
        ]
        dispatcher.utter_message(
            text=f"{offset_line}\nWhat would you like to do next?",
            buttons=buttons,
        )

        # A compact record of the plan, so a handover after the form still has
        # the context the form-scoped slots are about to lose.
        trip_summary = {
            "destination":          destination,
            "origin":               origin,
            "dates":                dates,
            "budget_eur":           budget,
            "sustainability_level": level,
            "transport_preference": preference,
            "trip_length":          trip_length,
            "distance_km":          round(distance) if distance else None,
            "recommended_mode":     best["mode"] if best else None,
            "recommended_carbon_kg": round(best["carbon_kg"], 1) if best else None,
            "hotels_shown":         [h["name"] for h in ranked],
        }

        # Keep destination and origin but clear the form-scoped inputs so a new
        # trip can be planned cleanly.
        return [
            SlotSet("last_trip_summary", json.dumps(trip_summary)),
            SlotSet("travel_dates", None),
            SlotSet("budget", None),
            SlotSet("sustainability_level", None),
            SlotSet("transport_preference", None),
            SlotSet("trip_length", None),
        ]


class ActionDefaultFallback(Action):
    """Override Rasa's built-in `action_default_fallback`.

    Rasa's built-in two-stage fallback (`action_two_stage_fallback`) runs its
    own affirm + rephrase loop. When both stages fail, it fires
    `action_default_fallback` — which by default just says "sorry" and stops.
    Here we override that to hand the conversation to a human advisor with
    full context, matching the brief's escalation requirement.
    """

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        advisor = random.choice(["Anna", "Marco", "Priya", "Ling", "Ines"])
        package = _build_handover_package(tracker, reason="two_stage_fallback_exhausted")

        print("=" * 60, flush=True)
        print("HUMAN HANDOVER PACKAGE (fallback exhausted)", flush=True)
        print(json.dumps(package, indent=2, default=str), flush=True)
        print("=" * 60, flush=True)

        dispatcher.utter_message(
            text=(
                f"🎫 HANDOVER · Ticket {package['ticket_id']} — I'm still not "
                f"following, so I'm passing you to travel advisor {advisor}. "
                "They will see the recent conversation and pick up where we left off.\n"
                f"For what it's worth: {SCOPE_SUMMARY}"
            )
        )
        # Deactivate the two-stage fallback loop and revert the unrecognised
        # user message so the fallback rule does not re-fire on the next turn.
        return [
            SlotSet("handover_active", True),
            ActiveLoop(None),
            UserUtteranceReverted(),
        ]


# One sentence describing what this bot is for, reused everywhere the bot has
# to admit it did not understand. Keep it in step with utter_ask_rephrase and
# utter_out_of_scope_help in domain.yml.
SCOPE_SUMMARY = (
    "I'm an eco-travel assistant: I can plan a sustainable trip, compare "
    "low-carbon ways to travel, estimate a journey's carbon footprint, and "
    "suggest places to stay and things to do."
)

INTENT_LABELS = {
    "greet":                 "say hello",
    "goodbye":               "say goodbye",
    "ask_weather":           "the weather somewhere",
    "ask_currency":          "an exchange rate",
    "plan_trip":             "plan a sustainable trip",
    "ask_eco_hotels":        "places to stay",
    "ask_green_transport":   "low-carbon ways to travel",
    "ask_carbon_footprint":  "the carbon footprint of a journey",
    "ask_local_activities":  "things to do that support locals",
    "ask_carbon_offset":     "carbon offset programmes",
    "request_human_advisor": "talk to a human advisor",
    "ask_place_description": "what a place is like",
    "bot_challenge":         "what this bot can do",
    "thank_you":             "say thanks",
    "off_topic":             "something outside eco-travel",
    "ask_booking":           "book something or see live prices",
    "ask_regulated_advice":  "visa, health or safety rules",
    "insult":                "tell me I got it wrong",
    "ask_privacy":           "what happens to your data",
    "stop":                  "stop the current questions",
    "inform":                "answer the question I asked",
}


class ActionDefaultAskAffirmation(Action):
    """Ask "did you mean…?" with readable options instead of intent names.

    Rasa's built-in version of this action prints the raw intent name —
    "Did you mean 'ask_carbon_offset'?" — which means nothing to a user. The
    candidates come from DIET's own intent_ranking for the message that
    failed, so the buttons are whatever the classifier thought was close.

    Clicking a candidate sends `/<intent>` as the payload, which the
    two-stage fallback loop treats as a successful clarification; "None of
    these" sends /out_of_scope, which moves it on to the rephrase stage.
    """

    def name(self) -> Text:
        return "action_default_ask_affirmation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        ranking = (tracker.latest_message or {}).get("intent_ranking") or []
        candidates = [
            r for r in ranking
            if r.get("name") not in ("nlu_fallback", "out_of_scope", None)
        ][:2]

        buttons = [
            {
                "title": INTENT_LABELS.get(
                    r["name"], r["name"].replace("_", " ")
                ).capitalize(),
                "payload": f"/{r['name']}",
            }
            for r in candidates
        ]
        # Always offer the bot's main job, in case none of DIET's guesses are
        # close and the user simply does not know what this bot is for.
        if not any(b["payload"] == "/plan_trip" for b in buttons):
            buttons.append({"title": "Plan a sustainable trip", "payload": "/plan_trip"})
        buttons.append({"title": "None of these", "payload": "/out_of_scope"})

        dispatcher.utter_message(
            text=(
                f"Sorry, I'm not sure what you meant. {SCOPE_SUMMARY}\n"
                "Were you asking about:"
            ),
            buttons=buttons,
        )
        return []


class ActionCoreFallback(Action):
    """What to do when the message was understood but no rule or story covers it.

    Configured as RulePolicy's `core_fallback_action_name`. This is a different
    situation from an NLU fallback: the classifier is confident ("What is
    2+2" -> `inform`, 0.94), but nothing in the rules says what `inform` means
    outside the trip form.

    It used to be `action_two_stage_fallback`, which froze the whole server.
    That action is a loop built for the NLU case: it activates, sees the last
    intent is not `nlu_fallback`, deactivates without saying anything, and the
    policies — still unsure — pick it again. One message ran it 255 times in
    25 seconds, copying the whole tracker each time, until the Streamlit
    request timed out.

    This action says something useful once and reverts the user's message, so
    the policies are not asked the same unanswerable question again, and any
    slots extracted from it (the "budget" of "2+2") are discarded.
    """

    def name(self) -> Text:
        return "action_core_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        dispatcher.utter_message(
            text=(
                "I'm not sure what to do with that one. "
                f"{SCOPE_SUMMARY}"
            ),
            buttons=[
                {"title": "Plan a sustainable trip", "payload": "/plan_trip"},
                {"title": "Low-carbon transport", "payload": "/ask_green_transport"},
                {"title": "Talk to a human advisor", "payload": "/request_human_advisor"},
            ],
        )
        return [UserUtteranceReverted()]


class ValidateTripPlanningForm(FormValidationAction):
    """Adaptive trip-planning form.

    - `required_slots()` decides which slot to ask for NEXT, based on prior
      answers. This is the "adaptive questioning" the brief calls for.
    - `validate_<slot>()` methods reject nonsense inputs — free-text slots
      need validation because buttons cannot cover every case. This is the
      brief's "error recovery mechanism".
    """

    def name(self) -> Text:
        return "validate_trip_planning_form"

    async def required_slots(
        self,
        domain_slots: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Text]:
        # domain_slots includes every slot the form MIGHT ask (declared in
        # domain.yml). This override filters that list down based on earlier
        # answers so we skip questions that are not relevant.
        # Never modify the given list itself (Making a Bot Behave §2.3 gotcha).
        slots = list(domain_slots)

        # Rule 1: transport_preference is only relevant if the user picked
        # HIGH sustainability. Skip it otherwise.
        sust = tracker.get_slot("sustainability_level")
        if sust is not None and sust != "high" and "transport_preference" in slots:
            slots.remove("transport_preference")

        # Rule 2: trip_length is only relevant on a tight budget (< 500 EUR).
        # Otherwise assume the user has time flexibility and skip the question.
        budget_raw = tracker.get_slot("budget")
        if budget_raw:
            digits = "".join(ch for ch in str(budget_raw) if ch.isdigit())
            if digits and int(digits) >= 500 and "trip_length" in slots:
                slots.remove("trip_length")

        return slots

    def validate_destination(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        text = str(slot_value or "").strip()
        if not text:
            dispatcher.utter_message(text="I need a destination — a city name works best.")
            return {"destination": None}
        if text[0].isdigit():
            dispatcher.utter_message(text="That doesn't look like a city name. Try 'Barcelona' or 'Kyoto'.")
            return {"destination": None}
        return self._check_place("destination", text, dispatcher, tracker)

    @staticmethod
    def _check_place(slot: Text, text: Text,
                     dispatcher: CollectingDispatcher,
                     tracker: Tracker) -> Dict[Text, Any]:
        """Shared checks for a free-text place answer inside the form.

        The place slots are filled `from_text`, so whatever the user types is
        offered as the answer — including "What is 2+2", which used to become
        the destination. Reject answers the classifier recognised as something
        else entirely, then confirm the place exists. A map service that is
        down must not block the form, so that case is accepted with a note.
        """
        intent = ((tracker.latest_message or {}).get("intent") or {}).get("name")
        if intent in ("off_topic", "bot_challenge"):
            question = ("Where would you like to travel?" if slot == "destination"
                        else "Where will you be travelling from?")
            dispatcher.utter_message(
                text=f"Let's finish planning first — I need a city. {question}"
            )
            return {slot: None}

        # Cities with pre-fetched data are known to exist; skip the API call.
        if (ECO_DATA_DIR / f"{text.lower()}_hotels.json").exists() \
                or text.lower() in LOCAL_ACTIVITIES:
            return {slot: text}

        coords, status = geocode_city_result(text)
        if coords is None and status == GEOCODE_NOT_FOUND:
            dispatcher.utter_message(
                text=f"I couldn't find a place called '{text}'. Could you try another spelling?"
            )
            return {slot: None}
        if coords is None:
            dispatcher.utter_message(
                text=(
                    f"I can't reach the map service to check '{text}' right now, "
                    "so I'll take your word for it — carbon figures for the journey "
                    "may be missing from the plan."
                )
            )
        return {slot: text}

    def validate_origin(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        text = str(slot_value or "").strip()
        if len(text) < 2 or text[0].isdigit():
            dispatcher.utter_message(
                text="Which city are you starting from? A city name works best."
            )
            return {"origin": None}
        # Geocode now rather than at plan time: a place we cannot find would
        # silently drop the whole transport comparison later on.
        return self._check_place("origin", text, dispatcher, tracker)

    def validate_travel_dates(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        text = str(slot_value or "").strip()
        if len(text) < 3:
            dispatcher.utter_message(
                text="I need dates or a rough time frame, e.g. '15-20 August' or 'next week'."
            )
            return {"travel_dates": None}
        # Reject pure numeric input — that is almost certainly a budget answer
        # typed against the wrong prompt.
        if text.isdigit():
            dispatcher.utter_message(
                text="That looks like a number, not a date. Try '15-20 August' or 'next week'."
            )
            return {"travel_dates": None}
        return {"travel_dates": text}

    def validate_budget(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        digits = "".join(ch for ch in str(slot_value or "") if ch.isdigit())
        if not digits:
            dispatcher.utter_message(text="I need a number for the budget, e.g. 800.")
            return {"budget": None}
        if int(digits) == 0:
            dispatcher.utter_message(text="The budget must be greater than zero.")
            return {"budget": None}
        return {"budget": digits}

    def validate_sustainability_level(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        text = str(slot_value or "").strip().lower()
        # Order matters: check specific tokens before generic ones.
        if any(k in text for k in ("high", "eco", "green", "sustainable")):
            return {"sustainability_level": "high"}
        if any(k in text for k in ("medium", "mid", "moderate", "middle")):
            return {"sustainability_level": "medium"}
        if any(k in text for k in ("low", "cheap", "budget")):
            return {"sustainability_level": "low"}

        dispatcher.utter_message(text="Please pick one: low, medium, or high.")
        return {"sustainability_level": None}

    def validate_transport_preference(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        text = str(slot_value or "").strip().lower()
        if "train" in text or "bus" in text or "train_or_bus" in text:
            return {"transport_preference": "train_or_bus"}
        if "any" in text or "low" in text or "any_low_carbon" in text:
            return {"transport_preference": "any_low_carbon"}
        dispatcher.utter_message(text="Please pick 'train or bus' or 'any low carbon'.")
        return {"transport_preference": None}

    def validate_trip_length(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        text = str(slot_value or "").strip().lower()
        if "weekend" in text:
            return {"trip_length": "weekend"}
        if "week" in text:
            return {"trip_length": "week"}
        if "extend" in text or "long" in text:
            return {"trip_length": "extended"}
        dispatcher.utter_message(text="Please pick weekend, one week, or extended.")
        return {"trip_length": None}
