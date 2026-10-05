from typing import Any, Text, Dict, List, Optional, Tuple
import json
import os
import re
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
# External API config (Nominatim / Overpass / Wikipedia)
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

# Volunteer-run APIs (Nominatim, Overpass) require a User-Agent that identifies
# the application ("Provide a valid HTTP Referer or User-Agent identifying the
# application", https://operations.osmfoundation.org/policies/nominatim/); the
# lecturer's Eco-Travel APIs README asks for a contact address as well. The
# default points to the public repository. A deployed instance should override
# ECO_USER_AGENT so complaints reach its operator.
USER_AGENT = os.environ.get(
    "ECO_USER_AGENT",
    "EcoTravelAdvisor/1.0 (+https://github.com/barislaca94/Sustainable-ChatBot)",
)
HTTP_HEADERS = {"User-Agent": USER_AGENT}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# In-memory cache for geocoded coordinates, so Nominatim is not called again
# for the same city on every user turn. Nominatim allows 1 req/sec — the sleep
# only applies when the API is actually called.
_GEOCODE_CACHE: Dict[str, Tuple[float, float]] = {}
_LAST_NOMINATIM_CALL: List[float] = [0.0]

# Pre-fetched Overpass POI data (see scripts/fetch_pois.py).
ECO_DATA_DIR = REPO_ROOT / "data" / "eco_data"


# Outcome of a geocoding attempt. "not_found" and "unavailable" must stay
# apart: telling a user that Berlin does not exist because Nominatim was rate
# limiting the bot is both wrong and, inside a form, a dead end — the slot gets
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

        city = tracker.get_slot("city_name") or place_in_context(tracker, dispatcher)[1]

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
# Eco-travel: hotel ranking from pre-fetched OSM data, activities and offsets
# =============================================================================

# NOTE: The old ECO_HOTELS dict fabricated eco-certifications for named
# properties. That is exactly the greenwashing pattern the assignment brief
# forbids, so it has been removed. Hotel data now comes from real OSM records
# in `data/eco_data/<city>_hotels.json` (pre-fetched by scripts/fetch_pois.py).
# The bot presents PROXY sustainability signals (transit proximity, room count,
# parking availability) and explicitly disclaims that these are not
# certifications. See the lecturer's Eco-Travel APIs README, "The gap you will have to think about".


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
    # Measured coverage across the cached cities: 0 of 297 hotels carry it, so
    # it does not contribute with the current data — kept because when present
    # it is solid.
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

    # No positive signal at all means OpenStreetMap is silent about this hotel,
    # not that it is a poor choice. Showing it red would turn missing data into
    # a negative claim (the lecturer's Eco-Travel APIs README: "Say nothing
    # about sustainability you cannot evidence"), so it gets a neutral band.
    if positives == 0:
        return positives, "⚪"
    if positives >= 1.5:
        return positives, "🟢"
    if positives >= 0.8:
        return positives, "🟡"
    return positives, "🔴"


def format_signals(signals: Dict[str, Any]) -> str:
    """Human-readable summary of the proxy signals recorded for a hotel."""
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
    "⚪": "Not enough OpenStreetMap data to judge",
}

# Nights implied by the trip_length slot, used to turn a total budget into a
# per-night figure for the star-rating price proxy.
TRIP_NIGHTS = {"weekend": 2, "week": 7, "extended": 14}

# Share of the total budget assumed to go on accommodation once travel,
# food and activities are accounted for. A rough planning heuristic, stated
# as such wherever it reaches the user.
ACCOMMODATION_BUDGET_SHARE = 0.5


_BUDGET_NUMBER = re.compile(r"(\d+(?:[.,]\d+)*)\s*(k)?\b", re.IGNORECASE)


def _to_number(raw: Text, thousands: bool) -> float:
    """'1,500' / '1.500' are thousands separators; '1.5' / '2,5' are decimals."""
    groups = re.split(r"[.,]", raw)
    if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]):
        value = float("".join(groups))
    else:
        value = float(raw.replace(",", "."))
    return value * 1000 if thousands else value


_UNIT_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS_WORDS = {w: 10 * i for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_THOUSAND_WORDS = {"thousand", "grand"}
_WORD_TOKEN = re.compile(r"[A-Za-z]+")


def _small_value(word: Text) -> Optional[int]:
    return _UNIT_WORDS.get(word, _TENS_WORDS.get(word))


def words_to_digits(text: Text) -> Text:
    """Replace English number words with digits: "about fifteen hundred
    euros" -> "about 1500 euros".

    Covers what people type for a budget: "fifteen hundred", "two thousand",
    "one thousand five hundred", "twelve hundred and fifty", "a thousand",
    "two grand", "twenty-five hundred". "and" joins a number only when a
    small number follows ("twelve hundred and fifty"); before another
    hundred or thousand it separates two numbers, so "between six hundred
    and seven hundred" stays a range.
    """
    tokens = list(_WORD_TOKEN.finditer(text))
    words = [t.group().lower() for t in tokens]
    spans: List[Tuple[int, int, int]] = []   # (start char, end char, value)
    i = 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        starts = _small_value(w) is not None or w == "hundred" or w in _THOUSAND_WORDS \
            or (w in ("a", "an") and (nxt == "hundred" or nxt in _THOUSAND_WORDS))
        if not starts:
            i += 1
            continue
        total = current = 0
        seen_scale = False
        start = tokens[i].start()
        j = i
        while j < len(words):
            w = words[j]
            small = _small_value(w)
            if w in ("a", "an") and j == i:
                current = 1
            elif small is not None:
                current += small
            elif w == "hundred":
                current = (current or 1) * 100
                seen_scale = True
            elif w in _THOUSAND_WORDS:
                total += (current or 1) * 1000
                current = 0
                seen_scale = True
            elif w == "and" and seen_scale and j + 1 < len(words):
                # Join "twelve hundred and fifty", but not "six hundred and
                # seven hundred": look at what the next number is made of.
                k = j + 1
                while k < len(words) and _small_value(words[k]) is not None:
                    k += 1
                if k == j + 1 or (k < len(words) and (words[k] == "hundred" or words[k] in _THOUSAND_WORDS)):
                    break
            else:
                break
            j += 1
        spans.append((start, tokens[j - 1].end(), total + current))
        i = j
    for start, end, value in reversed(spans):
        text = text[:start] + str(value) + text[end:]
    return text


def parse_budget(text: Any) -> Tuple[Optional[int], bool]:
    """Read a budget in EUR from free text.

    Returns (amount, is_range). A range such as "800 to 1000" or "800-1000"
    gives its midpoint (900) and is_range=True, so the caller can say which
    figure it used. "2k" means 2000. Only the first two numbers count, which
    keeps "1500 for 2 people" at 1500. Before this, every caller joined all
    the digits, so "800 to 1000" became 8001000.

    A message with no digits at all is first passed through words_to_digits,
    so a typed "about fifteen hundred euros" works like "about 1500 euros"
    instead of the form asking the same question again.
    """
    text = str(text or "")
    if not re.search(r"\d", text):
        text = words_to_digits(text)
    numbers = [_to_number(num, bool(k)) for num, k in _BUDGET_NUMBER.findall(text)]
    if not numbers:
        return None, False
    if len(numbers) >= 2 and re.search(r"\d\s*(?:-|–|to|and)\s*\d", str(text), re.IGNORECASE):
        return int(round((numbers[0] + numbers[1]) / 2)), True
    return int(round(numbers[0])), False


def nightly_budget(tracker: Tracker) -> Optional[float]:
    """Approximate EUR available per night, or None when unknowable.

    Needs both a budget and a trip length; without the second, any per-night
    figure would be invented, so the price proxy is simply not applied.
    """
    budget, _ = parse_budget(tracker.get_slot("budget"))
    nights = TRIP_NIGHTS.get(tracker.get_slot("trip_length") or "")
    if not budget or not nights:
        return None
    return budget * ACCOMMODATION_BUDGET_SHARE / nights


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


def supported_cities() -> List[str]:
    """Cities with pre-fetched hotel data, read from data/eco_data/ so the list
    the bot offers can never name a city it has no data for."""
    return sorted(p.name[: -len("_hotels.json")].title()
                  for p in ECO_DATA_DIR.glob("*_hotels.json"))


SUPPORTED_CITIES_HINT = ", ".join(supported_cities())

# LOCAL_ACTIVITIES is a short hand-written list, not a verified directory, so
# it is presented as examples of what to look for, not as recommendations of
# specific businesses (the lecturer's Eco-Travel APIs README on greenwashing).
ACTIVITIES_CAVEAT = (
    "Illustrative examples of the kind of activity to look for; check that they "
    "are still running and locally owned before you book."
)

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


_PLACE_AFTER_PREPOSITION = re.compile(
    r"\b(?:in|to|from|visit|visiting|near)\s+([A-Za-zÀ-ÿ' .\-]{2,40})$", re.IGNORECASE)


def place_from_answer(text: Text, entities: Optional[List[Dict[Text, Any]]] = None) -> Text:
    """The place inside a typed answer such as "I live in Berlin".

    Place questions in the form are filled from the whole reply (from_text),
    so a sentence used to be looked up as if it were a city name. In order:
    a place entity the NLU found in the reply, a city the bot has data for
    named in it, the words after a final "in / to / from"; otherwise the
    reply itself (a plain "Hallstatt" stays as it is).
    """
    lowered = text.lower()
    for e in entities or []:
        value = str(e.get("value") or "")
        if e.get("entity") in ("destination", "origin", "city_name") and value \
                and value.lower() in lowered:
            return value
    known = set(supported_cities()) | {c.title() for c in LOCAL_ACTIVITIES}
    for city in sorted(known, key=len, reverse=True):
        if re.search(rf"\b{re.escape(city.lower())}\b", lowered):
            return city
    match = _PLACE_AFTER_PREPOSITION.search(text.strip(" .!?"))
    return match.group(1).strip() if match else text


def has_activity_data(city: str) -> bool:
    """True when the activities answer has something to show for `city`,
    so no button offers activities the bot does not have."""
    return bool(LOCAL_ACTIVITIES.get((city or "").lower()) or nearby_cultural_sites(city))


# Offset schemes listed as starting points only. Names and addresses, no
# ratings: the earlier "verified", "backed by WWF" and "CDM-certified" notes
# had no source in this project, and the lecturer's Eco-Travel APIs README
# says to say nothing about sustainability that cannot be evidenced.
CARBON_OFFSET_PROGRAMS = [
    {"name": "Gold Standard", "url": "https://www.goldstandard.org"},
    {"name": "Atmosfair", "url": "https://www.atmosfair.de"},
    {"name": "Klima", "url": "https://klima.com"},
    {"name": "myclimate", "url": "https://www.myclimate.org"},
]

OFFSET_CAVEAT = (
    "Listed as starting points, not endorsements: this assistant has not "
    "checked their projects. Compare how each one certifies and reports its "
    "projects on its own site before you pay."
)


# kg CO2e per passenger-km, used whenever Climatiq is unavailable.
#
# Aligned on 2026-09-16 with the factors Climatiq serves for the activity ids
# below (scripts/climatiq_check.py), so that a fallback answer does not
# contradict a live one. Re-checked on 2026-10-05 against Climatiq data
# version ^21, which names its sources: train, coach and short-haul flight from
# the UK government (DESNZ/BEIS) "Greenhouse gas reporting: conversion factors
# 2026"; car from the German Environment Agency (UBA) emission factor list
# V2.1, 2024. The earlier table used a domestic-flight figure (0.255) for every
# flight, which roughly doubled short-haul estimates. Ferry, bicycle and walk
# are not covered by CLIMATIQ_ACTIVITY_IDS and keep the earlier table values.
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

LOCAL_SOURCE_LABEL = "local table of DESNZ 2026 and UBA 2024 factors"

# Words users write for a transport mode -> the mode name the carbon tables use
# "Emissions for driving to Prague" used to be priced as a flight,
# because "driving" was neither annotated nor mapped and the default was flight.
MODE_ALIASES = {
    "flight": "flight", "flights": "flight", "fly": "flight", "flying": "flight",
    "plane": "flight", "airplane": "flight", "aeroplane": "flight",
    "car": "car", "drive": "car", "driving": "car",
    "train": "train", "trains": "train", "rail": "train",
    "bus": "bus", "coach": "bus",
    "ferry": "ferry", "boat": "ferry",
    "bike": "bicycle", "bicycle": "bicycle", "cycling": "bicycle", "cycle": "bicycle",
    "walk": "walk", "walking": "walk",
}
_MODE_WORD = re.compile(r"\b(" + "|".join(sorted(MODE_ALIASES, key=len, reverse=True)) + r")\b")
COMPARED_MODES = ("train", "bus", "car", "flight")


def resolve_mode(slot_value: Any, text: Any) -> Optional[Text]:
    """The transport mode the user meant: the NLU entity first, then any mode
    word in the message itself, so a missed entity annotation does not fall
    back to a guess. None when neither names a mode."""
    if slot_value:
        mode = MODE_ALIASES.get(str(slot_value).lower().strip())
        if mode:
            return mode
    match = _MODE_WORD.search(str(text or "").lower())
    return MODE_ALIASES[match.group(1)] if match else None


# Words that end a place name inside "from X to Y" ("... to Paris next week").
_ROUTE_STOP = (
    r"by|on|in|for|next|this|tomorrow|today|tonight|via|with|and|please|"
    r"leaving|departing|around|at|during|or|is|would|which|what"
)
_PLACE = rf"((?:(?!(?:{_ROUTE_STOP})\b)[A-Za-zÀ-ÿ'.\-]+\s*){{1,3}})"
_ROUTE_PATTERNS = (
    re.compile(rf"\bfrom\s+{_PLACE}\s*\bto\s+{_PLACE}", re.IGNORECASE),
    re.compile(rf"\bbetween\s+{_PLACE}\s*\band\s+{_PLACE}", re.IGNORECASE),
)


def route_from_text(text: Any,
                    entities: Optional[List[Dict[Text, Any]]] = None
                    ) -> Optional[Tuple[Text, Text]]:
    """(origin, destination) when the message itself says "from X to Y" or
    "between X and Y", else None.

    The NLU entities can mislabel a city ("green transport from London to
    Paris" once came back with Paris as a second origin), and a destination
    left in a slot by an earlier turn would then be used silently. When the
    user names the route in this very message, its word order decides which
    place is the origin and which the destination, the same idea as
    resolve_mode() for the transport mode.

    The place names themselves come from the NLU entities when one lies
    inside each part of the route ("to Paris sustainably" -> "Paris"); the
    raw words are used only when no entity was found there.
    """
    raw = str(text or "")
    if raw.startswith("/"):
        return None   # a button payload sets the slots itself
    places = [str(e.get("value")) for e in (entities or [])
              if e.get("entity") in ("origin", "destination", "city_name") and e.get("value")]
    for pattern in _ROUTE_PATTERNS:
        match = pattern.search(raw)
        if not match:
            continue
        found = []
        for part in match.groups():
            part = part.strip(" .,'-")
            inside = [p for p in places if p.lower() in part.lower()]
            found.append(max(inside, key=len) if inside else part)
        origin, destination = found
        if origin and destination and origin.lower() != destination.lower():
            return origin, destination
    return None


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
    """Carbon estimate from the built-in factor table (EMISSION_FACTORS)."""
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

    # Serve modes already looked up in this process.
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


def approx_distance_result(origin: str, destination: str) -> Tuple[float, Text]:
    """Distance in km plus the geocoding status, so callers can explain a zero.

    Status is GEOCODE_OK, GEOCODE_NOT_FOUND (a place that genuinely cannot be
    found) or GEOCODE_UNAVAILABLE (the map service is down or rate limiting).
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
# station access). Rough planning assumptions chosen for this prototype;
# no published source gives these figures for all modes, so every answer
# labels them as estimates.
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
# as price; a budget-first user ("low") gets cost weighted 0.7 against 0.3 for
# carbon.
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

    budget_eur, _ = parse_budget(budget)

    options: List[Dict[Text, Any]] = []
    for mode in kept:
        # Min-max normalise within this comparison, then invert so that 1.0 is
        # always "best on this axis". When every candidate ties on an axis (or
        # there is only one), each gets the best value, 1.0, on it.
        c_norm = 0.0 if c_hi == c_lo else (carbon[mode] - c_lo) / (c_hi - c_lo)
        p_norm = 0.0 if p_hi == p_lo else (costs[mode] - p_lo) / (p_hi - p_lo)
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
    "Costs are rough per-kilometre planning assumptions, not live fares — no fare "
    "source is connected (Amadeus, the planned one, was decommissioned). Treat "
    "them as an order of magnitude and check a booking site before deciding."
)


def _last_trip(tracker: Tracker) -> Dict[Text, Any]:
    """The last completed trip plan (last_trip_summary), or {}."""
    raw = tracker.get_slot("last_trip_summary")
    try:
        trip = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {}
    return trip if isinstance(trip, dict) else {}


def place_in_context(tracker: Tracker,
                     dispatcher: CollectingDispatcher) -> Tuple[Optional[Text], Optional[Text]]:
    """(origin, destination) for a request that may not name the place.

    The trip form clears its slots on submit, so that a second plan asks every
    question again (Making a Bot Behave §6.3). The brief also asks for the
    destination to persist across turns, so a follow-up such as "and hotels?"
    falls back to the last completed plan, and the bot says so, which lets
    the user correct it. A place named in the message or still in a slot
    always wins. The origin is taken from the plan only when the destination
    is the plan's own.
    """
    origin = tracker.get_slot("origin")
    destination = tracker.get_slot("destination") or tracker.get_slot("city_name")
    trip = _last_trip(tracker)
    planned = trip.get("destination")
    if not destination and planned:
        destination = planned
        dispatcher.utter_message(text=f"ℹ️ Using {planned.title()} from your trip plan.")
    if not origin and planned and destination and destination.lower() == planned.lower():
        origin = trip.get("origin")
    return origin, destination


def trip_preferences(tracker: Tracker,
                     destination: Optional[Text]) -> Tuple[Optional[Text], Optional[Text], Any, bool]:
    """(sustainability level, transport preference, budget, from_plan).

    The form slots win while they are set. The form clears them on submit,
    so a follow-up about the planned destination (the plan's own "Green
    transport" button, or "and how do I get there?") falls back to the
    answers kept in last_trip_summary; from_plan tells the caller to say so.
    """
    level = tracker.get_slot("sustainability_level")
    preference = tracker.get_slot("transport_preference")
    budget = tracker.get_slot("budget")
    if level or preference or budget:
        return level, preference, budget, False
    trip = _last_trip(tracker)
    planned = str(trip.get("destination") or "")
    if planned and destination and planned.lower() == str(destination).lower():
        return (trip.get("sustainability_level"), trip.get("transport_preference"),
                trip.get("budget_eur"), True)
    return None, None, None, False


OVERLAND_MODES = ("train", "bus", "coach", "car")


def overland_caveat(distance_km: float, modes: List[Text]) -> Optional[Text]:
    """A warning for long routes that still list train, bus or car.

    Distances are great-circle and no route planner is connected, so the bot
    cannot tell whether a rail or road connection exists (London to Kyoto
    still produced a bus). The cut-off reuses the long-haul boundary of the
    flight emission factors (CLIMATIQ_SHORT_HAUL_KM, 3,700 km).
    """
    if distance_km < CLIMATIQ_SHORT_HAUL_KM:
        return None
    if not any(m.lower() in OVERLAND_MODES for m in modes):
        return None
    return (
        f"⚠️ About {distance_km:,.0f} km in a straight line: I don't check whether a "
        "rail, bus or road route exists, so treat the overland options as rough "
        "comparisons, not suggestions."
    )


# =============================================================================
# Eco-travel actions
# =============================================================================

class ActionSuggestEcoHotels(Action):
    """List hotels in the requested city, ranked by proxy sustainability
    signals from OpenStreetMap. Never claims a hotel is 'eco-certified' —
    OSM has no reliable certification tags (the lecturer's Eco-Travel APIs README).
    """

    def name(self) -> Text:
        return "action_suggest_eco_hotels"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        _, destination = place_in_context(tracker, dispatcher)

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
                    f"I don't have hotel data for {destination} yet, so I can't rank places "
                    f"to stay there. Cities I do have data for: {SUPPORTED_CITIES_HINT}."
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

        _, destination = place_in_context(tracker, dispatcher)
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
        if has_activity_data(destination):
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

        latest = tracker.latest_message or {}
        route = route_from_text(latest.get("text"), latest.get("entities"))
        # A route named in the message wins; only without one is the context
        # (slots, then the last trip plan) consulted.
        origin, destination = route or place_in_context(tracker, dispatcher)

        if not origin or not destination:
            dispatcher.utter_message(
                text="Please tell me both the origin and destination. Example: 'green transport from London to Paris'."
            )
            return []
        # Keep the slots in step with the route actually used, so a follow-up
        # ("and the carbon footprint?") does not fall back to stale values.
        route_events = [SlotSet("origin", origin), SlotSet("destination", destination)] if route else []

        distance, geo_status = approx_distance_result(origin, destination)
        if distance == 0.0:
            dispatcher.utter_message(
                text=MAP_UNAVAILABLE_MESSAGE if geo_status == GEOCODE_UNAVAILABLE
                else f"Sorry, I couldn't locate {origin} or {destination} to estimate distance."
            )
            return []

        # Preferences come from the trip form while it is filled, or from the
        # last completed plan when the question is about its destination;
        # otherwise the weights default to "medium" and nothing is filtered out.
        level, preference, budget, from_plan = trip_preferences(tracker, destination)
        options, excluded, source_label = score_transport_options(
            distance,
            sustainability_level=level,
            transport_preference=preference,
            budget=budget,
        )

        header = [f"{origin.title()} → {destination.title()}: about {distance:.0f} km."]
        if from_plan:
            header.append("Using the preferences from your trip plan.")
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

        # On a long route the overland options are unchecked, so none is
        # marked as recommended (see overland_caveat).
        caveat = overland_caveat(distance, [o["mode"] for o in options])

        # One message per option so the UI can colour each as its own card.
        for index, option in enumerate(options):
            dispatcher.utter_message(
                text=format_transport_option(option, recommended=(index == 0 and not caveat))
            )

        if caveat:
            dispatcher.utter_message(text=caveat)

        # Alert on the high-emission option, quantified against the best one
        # (not on a long route, where the low-carbon option may not exist).
        worst = max(options, key=lambda o: o["carbon_kg"])
        best = min(options, key=lambda o: o["carbon_kg"])
        if worst["band"] == "🔴" and worst is not best and not caveat:
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
        return route_events


class ActionCalculateCarbon(Action):

    def name(self) -> Text:
        return "action_calculate_carbon"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        latest = tracker.latest_message or {}
        mode = resolve_mode(tracker.get_slot("transport_mode"), latest.get("text"))
        route = route_from_text(latest.get("text"), latest.get("entities"))
        origin, destination = route or place_in_context(tracker, dispatcher)

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

        # The slot is filled by a global from_entity mapping, so it would leak
        # into the next question; it is cleared after every answer.
        clear = [SlotSet("transport_mode", None)]
        if route:
            clear += [SlotSet("origin", origin), SlotSet("destination", destination)]

        if mode is None:
            # No mode named: compare the usual options instead of guessing one.
            carbon, source_label = estimate_carbon(list(COMPARED_MODES), distance)
            lines = [f"{origin.title()} to {destination.title()} ({distance:.0f} km), "
                     "estimated kg CO2e per passenger:"]
            for m in sorted(COMPARED_MODES, key=lambda m: carbon[m]):
                band = intensity_band(carbon[m], distance)
                lines.append(f"{band} {m.title()}: about {carbon[m]:.1f} kg — "
                             f"{BAND_LABELS.get(band, '')}")
            lines.append(source_label)
            dispatcher.utter_message(text="\n".join(lines))
            return clear

        carbon, source_label = estimate_carbon([mode], distance)
        kg = carbon[mode]
        band = intensity_band(kg, distance)
        reply = (
            f"{band} {mode.title()} from {origin.title()} to "
            f"{destination.title()} ({distance:.0f} km): about {kg:.1f} kg CO2e "
            f"per passenger — {BAND_LABELS.get(band, '')}.\n"
            f"That is {kg / distance:.3f} kg per passenger-km; the colour bands "
            f"are under {INTENSITY_GREEN} green, under {INTENSITY_AMBER} amber, "
            f"above that red.\n"
            f"{source_label}"
        )
        dispatcher.utter_message(text=reply)
        return clear


class ActionSuggestActivities(Action):

    def name(self) -> Text:
        return "action_suggest_activities"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        _, destination = place_in_context(tracker, dispatcher)

        if not destination:
            dispatcher.utter_message(text="Which city are you interested in?")
            return []

        acts = LOCAL_ACTIVITIES.get(destination.lower())
        sites = nearby_cultural_sites(destination)

        if not acts and not sites:
            dispatcher.utter_message(
                text=(
                    f"I don't have activity ideas for {destination} yet. "
                    f"Cities I do have them for: {', '.join(sorted(c.title() for c in LOCAL_ACTIVITIES))}."
                )
            )
            return []

        if acts:
            # No colour band: there is no carbon figure behind this list.
            lines = [f"Community-friendly, low-impact activities in {destination.title()}:"]
            lines.extend(f"• {a}" for a in acts)
            lines.append(ACTIVITIES_CAVEAT)
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

        lines = ["Carbon offset schemes you could look at:"]
        for p in CARBON_OFFSET_PROGRAMS:
            lines.append(f"• {p['name']} ({p['url']})")
        lines.append(OFFSET_CAVEAT)
        lines.append(
            "Ethical caveat: offsets should complement, not replace, choosing low-emission travel in the first place."
        )
        dispatcher.utter_message(text="\n".join(lines))
        return []


def _build_transcript(tracker: Tracker, turns: int = 10) -> List[str]:
    """The last `turns` user turns, each followed by the bot's replies to it.

    Counted in user turns, not messages: one long trip plan is several bot
    messages, and a message limit let a single plan push the user's own words
    out of the advisor's view. The bot's replies to a turn are joined on one
    line ("BOT : first | second") so each turn stays readable.
    """
    pairs: List[List[Text]] = []   # [user text, bot text, bot text, ...]
    for event in tracker.events:
        text = event.get("text")
        if not text:
            continue
        if event.get("event") == "user":
            pairs.append([text])
        elif event.get("event") == "bot":
            if not pairs:
                pairs.append([""])   # bot spoke first (session start)
            pairs[-1].append(text.replace("\n", " / "))
    transcript: List[str] = []
    for user_text, *bot_texts in pairs[-turns:]:
        if user_text:
            transcript.append(f"USER: {user_text}")
        if bot_texts:
            transcript.append("BOT : " + " | ".join(bot_texts))
    return transcript


def _build_handover_package(tracker: Tracker, reason: Text = "user_requested") -> Dict[Text, Any]:
    """Package the full conversation context for a human advisor handover.

    Follows the assignment brief: "Packaging full conversation context for
    human advisor handover" (Making a Bot Behave §5.3).
    """
    latest_intent = (tracker.latest_message or {}).get("intent", {}) or {}
    transcript = _build_transcript(tracker, turns=10)
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
    system would forward it to a ticketing system (for example Zendesk or an
    e-mail queue); no such integration is built. The message to the user says
    exactly that: it names no advisor and promises no response time, because
    no human is connected in this build ("Describing an integration you did
    not build is perfectly legitimate. Implying you built it is not",
    Making a Bot Behave §5.3).
    """

    def name(self) -> Text:
        return "action_human_handover"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        package = _build_handover_package(tracker, reason="user_requested")

        # Print the full package for the advisor pipeline (demo stub).
        print("=" * 60, flush=True)
        print("HUMAN HANDOVER PACKAGE (user_requested)", flush=True)
        print(json.dumps(package, indent=2, default=str), flush=True)
        print("=" * 60, flush=True)

        lines = [
            f"🎫 Ticket {package['ticket_id']} has been logged for a human travel advisor, "
            "with the context below.",
            "(Demo build: the package is written to the server log; a live deployment would post it to a ticketing system.)",
            "Context for the advisor:",
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
        distance, distance_status = (
            approx_distance_result(origin, destination) if origin else (0.0, GEOCODE_OK)
        )
        if distance:
            travel_options, excluded, carbon_source = score_transport_options(
                distance,
                sustainability_level=level,
                transport_preference=preference,
                budget=budget,
            )

        best = travel_options[0] if travel_options else None
        long_route_caveat = (overland_caveat(distance, [o["mode"] for o in travel_options])
                             if travel_options else None)
        # The card colour comes from the carbon figure of the recommended mode
        # (brief: the card is "driven by the carbon score"). With no figure —
        # no origin, or the map service down — the card is a neutral info box,
        # not a green one that would suggest a low-emission result.
        header_band = best["band"] if best and not long_route_caveat else "ℹ️"
        # The band word goes with the colour (never colour alone); the info
        # box has no band to name.
        header_label = f"{BAND_LABELS[header_band]} · " if header_band in BAND_LABELS else ""

        summary = [
            f"{header_band} {header_label}Trip plan — {destination.title()}"
            + (f" from {origin.title()}" if origin else ""),
            f"Dates: {dates}"
            + (f" · {trip_length} ({nights} nights)" if nights else "")
            + (f" · budget ~{budget} EUR" if budget else ""),
            f"Sustainability priority: {level}"
            + (f" · transport preference: {preference.replace('_', ' ')}" if preference else ""),
        ]
        if best and not long_route_caveat:
            summary.append(
                f"Recommended way to get there: {best['mode']} "
                f"(~{best['carbon_kg']:.0f} kg CO2e, ~{best['cost_eur']:.0f} EUR one way)."
            )
        dispatcher.utter_message(text="\n".join(summary))

        # ---- Transport cards ----
        if travel_options:
            for index, option in enumerate(travel_options[:3]):
                dispatcher.utter_message(
                    text=format_transport_option(
                        option, recommended=(index == 0 and not long_route_caveat))
                )
            if long_route_caveat:
                dispatcher.utter_message(text=long_route_caveat)
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
        elif distance_status == GEOCODE_UNAVAILABLE:
            # Say why the comparison is missing instead of silently dropping it.
            dispatcher.utter_message(
                text="ℹ️ I couldn't reach the map service just now, so I can't compare "
                     "transport options for this trip. Ask me again in a moment."
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
            lines.append(ACTIVITIES_CAVEAT)
            dispatcher.utter_message(text="\n".join(lines))

        # Say what is missing rather than leaving sections out silently.
        missing = [label for label, found in (("places to stay", ranked),
                                              ("activity ideas", acts)) if not found]
        if missing:
            dispatcher.utter_message(
                text=(
                    f"ℹ️ I don't have {' or '.join(missing)} for {destination.title()} yet. "
                    f"Cities with hotel data: {SUPPORTED_CITIES_HINT}. A human advisor "
                    "can help with other places."
                )
            )

        top_offset = CARBON_OFFSET_PROGRAMS[0]
        offset_line = (f"If you offset, compare schemes first, for example "
                       f"{top_offset['name']} ({top_offset['url']}) — not an endorsement.")
        if best:
            offset_line += (
                " Offsetting is a last step, though — the mode you pick matters more."
            )

        buttons = [
            {"title": f"Green transport to {destination.title()}",
             # Origin travels in the payload: the form slots are cleared below,
             # and without it the button would ask for the origin again.
             "payload": json_payload("ask_green_transport",
                                     destination=destination, origin=origin)},
            {"title": "Talk to a human advisor",
             "payload": "/request_human_advisor"},
        ]
        if has_activity_data(destination):
            buttons.insert(1, {"title": f"Community activities in {destination.title()}",
                               "payload": f'/ask_local_activities{{"destination":"{destination}"}}'})
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

        # Clear every form slot, destination and origin included: a filled slot
        # is skipped by the form, so a second plan would silently reuse the
        # first trip's cities ("Slots not cleared after the form → second run
        # skips questions", Making a Bot Behave §6.3). The plan itself survives
        # in last_trip_summary.
        return [
            SlotSet("last_trip_summary", json.dumps(trip_summary)),
            SlotSet("destination", None),
            SlotSet("origin", None),
            SlotSet("travel_dates", None),
            SlotSet("budget", None),
            SlotSet("sustainability_level", None),
            SlotSet("transport_preference", None),
            SlotSet("trip_length", None),
        ]


def json_payload(intent: Text, **entities: Any) -> Text:
    """A button payload that sets entities, e.g. /intent{"destination": "Kyoto"}.
    Entities with no value are left out."""
    values = {k: v for k, v in entities.items() if v}
    return f"/{intent}{json.dumps(values)}" if values else f"/{intent}"


TRIP_FORM_SLOTS = ("destination", "origin", "travel_dates", "budget",
                   "sustainability_level", "transport_preference", "trip_length")


class ActionCancelTripForm(Action):
    """Leave the trip form cleanly when the user says stop.

    The rule deactivates the form first; this action then clears every form
    slot and `requested_slot`, so a later "plan a trip" starts from the first
    question instead of resuming a half-filled form.
    """

    def name(self) -> Text:
        return "action_cancel_trip_form"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        dispatcher.utter_message(
            text="Okay, I've stopped the trip planning and cleared your answers. "
                 "What would you like to do instead?",
            buttons=[
                {"title": "Plan a new trip", "payload": "/plan_trip"},
                {"title": "Low-carbon transport", "payload": "/ask_green_transport"},
                {"title": "Talk to a human advisor", "payload": "/request_human_advisor"},
            ],
        )
        return [SlotSet(slot, None) for slot in TRIP_FORM_SLOTS] + [
            SlotSet("requested_slot", None)
        ]


class ActionAskTripPlanningFormDestination(Action):
    """Ask for the destination with one button per city that has hotel data.

    The brief asks for "quick-reply buttons generated dynamically from custom
    action responses for destination and preference selection". The cities
    come from data/eco_data/ (supported_cities), so a button never offers a
    city the bot cannot plan. The payload is the plain city name: it goes
    through the form's from_text mapping and validate_destination exactly as
    a typed answer would.
    """

    def name(self) -> Text:
        return "action_ask_trip_planning_form_destination"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        cities = supported_cities()
        dispatcher.utter_message(
            text="Great — where would you like to travel? Pick a city I have hotel "
                 "data for, or type any other city.",
            buttons=[{"title": city, "payload": city} for city in cities],
        )
        return []


def _trip_to(tracker: Tracker) -> Text:
    """' to Lisbon' when the destination is known, else '' — so a question can
    refer back to the answer the user has already given."""
    destination = str(tracker.get_slot("destination") or "").strip()
    return f" to {destination.title()}" if destination else ""


class ActionAskTripPlanningFormSustainabilityLevel(Action):
    """Ask how much sustainability matters, with one button per level.

    The preference questions are asked from Python for the same reason as the
    destination question: the brief asks for buttons "generated dynamically
    from custom action responses for destination and preference selection",
    and Making a Bot Behave §3.2 says the Python version "is what the brief
    actually asks for". The text names the options as well, because "people
    type instead of clicking" (§3.3); validate_sustainability_level reads the
    typed words.
    """

    def name(self) -> Text:
        return "action_ask_trip_planning_form_sustainability_level"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        dispatcher.utter_message(
            text=f"How important is sustainability for your trip{_trip_to(tracker)}? "
                 "Low, medium or high?",
            buttons=[
                {"title": "Low", "payload": '/inform{"sustainability_level":"low"}'},
                {"title": "Medium", "payload": '/inform{"sustainability_level":"medium"}'},
                {"title": "High", "payload": '/inform{"sustainability_level":"high"}'},
            ],
        )
        return []


class ActionAskTripPlanningFormTransportPreference(Action):
    """Asked only after "high" sustainability (required_slots, rule 1)."""

    def name(self) -> Text:
        return "action_ask_trip_planning_form_transport_preference"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        # "No flights" is what any_low_carbon does in _filter_modes.
        dispatcher.utter_message(
            text="You picked high sustainability — how do you want to get around? "
                 "Train or bus only, or any low-carbon option (no flights)?",
            buttons=[
                {"title": "Train or bus only",
                 "payload": '/inform{"transport_preference":"train_or_bus"}'},
                {"title": "Any low-carbon option",
                 "payload": '/inform{"transport_preference":"any_low_carbon"}'},
            ],
        )
        return []


class ActionAskTripPlanningFormTripLength(Action):
    """Asked only on a budget under 500 EUR (required_slots, rule 2)."""

    def name(self) -> Text:
        return "action_ask_trip_planning_form_trip_length"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        dispatcher.utter_message(
            text=f"With that budget, how long a trip{_trip_to(tracker)} do you have "
                 "in mind? A weekend, one week, or longer?",
            buttons=[
                {"title": "Weekend", "payload": '/inform{"trip_length":"weekend"}'},
                {"title": "One week", "payload": '/inform{"trip_length":"week"}'},
                {"title": "Extended", "payload": '/inform{"trip_length":"extended"}'},
            ],
        )
        return []


class ActionDefaultFallback(Action):
    """Override Rasa's built-in `action_default_fallback`.

    Rasa's built-in two-stage fallback (`action_two_stage_fallback`) runs its
    own affirm + rephrase loop. When both stages fail, it fires
    `action_default_fallback` — which by default just says "sorry" and stops.
    This override hands the conversation to a human advisor with full
    context instead, matching the brief's escalation requirement.
    """

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        package = _build_handover_package(tracker, reason="two_stage_fallback_exhausted")

        print("=" * 60, flush=True)
        print("HUMAN HANDOVER PACKAGE (fallback exhausted)", flush=True)
        print(json.dumps(package, indent=2, default=str), flush=True)
        print("=" * 60, flush=True)

        dispatcher.utter_message(
            text=(
                f"🎫 Ticket {package['ticket_id']} — I'm still not following, so I've "
                "logged this conversation for a human travel advisor. "
                "(Demo build: the package is written to the server log; a live deployment would post it to a ticketing system.)\n"
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


# Month names and relative dates. A date typed against the budget question
# ("15-20 August") otherwise parses as a 15-20 EUR range. "may" counts only
# next to a day number, so "I may spend 500" stays a budget.
_DATE_WORDS = re.compile(
    r"\b(january|february|march|april|june|july|august|september|october|november|"
    r"december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
    r"|\b\d{1,2}(st|nd|rd|th)?\s+may\b|\bmay\s+\d{1,2}\b"
    r"|\bnext (week|month)\b|\btomorrow\b|\btoday\b"
)


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
        # answers, so questions that are not relevant are skipped.
        # Never modify the given list itself (Making a Bot Behave §2.3 gotcha).
        slots = list(domain_slots)

        # Rule 1: transport_preference is only relevant if the user picked
        # HIGH sustainability. Skip it otherwise.
        sust = tracker.get_slot("sustainability_level")
        if sust is not None and sust != "high" and "transport_preference" in slots:
            slots.remove("transport_preference")

        # Rule 2: trip_length is only relevant on a tight budget (< 500 EUR).
        # Otherwise assume the user has time flexibility and skip the question.
        budget, _ = parse_budget(tracker.get_slot("budget"))
        if budget is not None and budget >= 500 and "trip_length" in slots:
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
        text = place_from_answer(text, (tracker.latest_message or {}).get("entities"))
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
        latest = tracker.latest_message or {}
        # At activation Rasa validates slots that are already filled
        # (rasa/core/actions/forms.py, FormAction.activate); no slot has been
        # requested yet. A place filled by an earlier question ("green transport
        # from Paris to Barcelona") but not named in the message that starts the
        # plan is not an answer to this form, so it is asked again instead of
        # being used silently (Making a Bot Behave §6.3: "Second run skips
        # questions"; Worksheet 3: "slot stores last extracted value across turns").
        if tracker.get_slot("requested_slot") is None \
                and text.lower() not in str(latest.get("text") or "").lower():
            return {slot: None}

        intent = (latest.get("intent") or {}).get("name")
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
        if text:
            text = place_from_answer(text, (tracker.latest_message or {}).get("entities"))
        if len(text) < 2 or text[0].isdigit():
            dispatcher.utter_message(
                text="Which city are you starting from? A city name works best."
            )
            return {"origin": None}
        destination = str(tracker.get_slot("destination") or "").strip()
        if destination and text.lower() == destination.lower():
            dispatcher.utter_message(
                text=f"That's the same as your destination, {destination}. "
                     "Where will you be travelling from?"
            )
            return {"origin": None}
        # Geocode now rather than at plan time: a place that cannot be found would
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
        if _DATE_WORDS.search(str(slot_value or "").lower()):
            dispatcher.utter_message(
                text="That looks like a date, not a budget. About how much do you want "
                     "to spend in EUR? (e.g. 800)"
            )
            return {"budget": None}
        budget, is_range = parse_budget(slot_value)
        if budget is None:
            dispatcher.utter_message(text="I need a number for the budget, e.g. 800.")
            return {"budget": None}
        if budget == 0:
            dispatcher.utter_message(text="The budget must be greater than zero.")
            return {"budget": None}
        if is_range:
            dispatcher.utter_message(text=f"I'll use about {budget} EUR, the middle of that range.")
        return {"budget": str(budget)}

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
