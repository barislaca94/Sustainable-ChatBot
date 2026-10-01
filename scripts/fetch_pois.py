"""Pre-fetch OpenStreetMap POI data for each supported destination.

The bot's actions server MUST NOT hit Overpass live — the API can be slow or
time out (30 s+), which violates the "under three seconds for critical
interactions" requirement in the assignment brief. Instead, run this script
once whenever the supported city list changes; the cached JSON files under
`data/eco_data/<city>_(hotels|transit|attractions).json` are read by
`actions/actions.py` at conversation time.

Both Nominatim (geocoding) and Overpass (POI query) are volunteer-funded and
require an identifying User-Agent. Nominatim rate-limits at 1 req/sec.

Usage:
    python scripts/fetch_pois.py                # all supported cities
    python scripts/fetch_pois.py Barcelona Berlin  # subset
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Override with ECO_USER_AGENT when running this from a shared/deployed host.
USER_AGENT = os.environ.get(
    "ECO_USER_AGENT",
    "EcoTravelAdvisor/1.0 (barislaca94@gmail.com)",
)
HEADERS = {"User-Agent": USER_AGENT}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "eco_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_CITIES = [
    "Barcelona",
    "Amsterdam",
    "Kyoto",
    "Lisbon",
    "Berlin",
    "Paris",
    "Copenhagen",
    "Oslo",
]

# (radius_metres, output_limit) per POI kind
KIND_CONFIG: Dict[str, Tuple[int, int]] = {
    "hotels":      (3000, 50),
    "transit":     (2000, 30),
    "attractions": (3000, 40),
}


def build_overpass_query(kind: str, lat: float, lon: float, radius: int, limit: int) -> str:
    """Return an Overpass QL query string for the given POI kind."""
    if kind == "hotels":
        body = f'nwr["tourism"="hotel"](around:{radius},{lat},{lon});'
    elif kind == "transit":
        body = (
            f'nwr["railway"="station"](around:{radius},{lat},{lon});'
            f'nwr["public_transport"="stop_position"](around:{radius},{lat},{lon});'
        )
    elif kind == "attractions":
        body = (
            f'nwr["tourism"="museum"](around:{radius},{lat},{lon});'
            f'nwr["tourism"="attraction"](around:{radius},{lat},{lon});'
            f'nwr["historic"="castle"](around:{radius},{lat},{lon});'
            f'nwr["historic"="monument"](around:{radius},{lat},{lon});'
        )
    else:
        raise ValueError(f"unknown kind: {kind}")
    return f"[out:json][timeout:60];({body});out center {limit};"


def geocode(city: str) -> Optional[Tuple[float, float]]:
    """Nominatim geocoding with a 1 req/sec pause afterwards."""
    print(f"  geocoding {city} …", flush=True)
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": city, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        data = response.json()
    except requests.RequestException as exc:
        print(f"    ! request failed: {exc}", flush=True)
        return None
    finally:
        time.sleep(1.1)  # respect Nominatim usage policy

    if not data:
        print("    ! no results", flush=True)
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def query_overpass(query: str, retries: int = 2) -> Optional[List[Dict[str, Any]]]:
    """Call Overpass API and return the parsed elements list."""
    for attempt in range(1, retries + 2):
        try:
            response = requests.post(
                OVERPASS_URL,
                data=query,
                headers=HEADERS,
                timeout=90,
            )
        except requests.RequestException as exc:
            print(f"    ! request failed on attempt {attempt}: {exc}", flush=True)
            time.sleep(3 * attempt)
            continue

        if response.status_code == 200:
            try:
                payload = response.json()
                return payload.get("elements", [])
            except json.JSONDecodeError:
                print(f"    ! non-JSON response on attempt {attempt}", flush=True)
        elif response.status_code == 429:
            print(f"    ! 429 rate limited on attempt {attempt}", flush=True)
        elif response.status_code == 504:
            print(f"    ! 504 gateway timeout on attempt {attempt}", flush=True)
        else:
            print(f"    ! HTTP {response.status_code} on attempt {attempt}", flush=True)

        time.sleep(5 * attempt)
    return None


def normalise_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reduce raw Overpass elements to a compact record for the bot to read."""
    out: List[Dict[str, Any]] = []
    for e in elements:
        tags = e.get("tags") or {}
        name = tags.get("name") or tags.get("name:en")
        if not name:
            continue

        lat = e.get("lat")
        lon = e.get("lon")
        if lat is None and "center" in e:
            lat = e["center"].get("lat")
            lon = e["center"].get("lon")
        if lat is None or lon is None:
            continue

        out.append({
            "id":   f"{e.get('type', 'node')}/{e.get('id')}",
            "name": name,
            "lat":  lat,
            "lon":  lon,
            "tags": tags,
        })
    return out


def fetch_city(city: str) -> None:
    """Geocode a city and fetch all POI kinds for it, writing JSON files."""
    print(f"[{city}]", flush=True)
    coords = geocode(city)
    if not coords:
        print(f"  skipping {city}: no coordinates", flush=True)
        return
    lat, lon = coords
    print(f"  centre: {lat:.4f}, {lon:.4f}", flush=True)

    for kind, (radius, limit) in KIND_CONFIG.items():
        out_path = OUT_DIR / f"{city.lower()}_{kind}.json"
        print(f"  querying Overpass · {kind} (r={radius}m, ≤{limit}) …", flush=True)
        query = build_overpass_query(kind, lat, lon, radius, limit)
        elements = query_overpass(query)
        if elements is None:
            print(f"    ! failed after retries — {kind} skipped", flush=True)
            continue

        records = normalise_elements(elements)
        out_path.write_text(json.dumps({
            "city":   city,
            "centre": {"lat": lat, "lon": lon},
            "kind":   kind,
            "radius_m": radius,
            "count":  len(records),
            "records": records,
        }, indent=2, ensure_ascii=False))
        print(f"    → {out_path.relative_to(REPO_ROOT)}  ({len(records)} records)", flush=True)

        # Be polite between Overpass calls even though the API is public.
        time.sleep(2)


def main(argv: List[str]) -> int:
    cities = [c for c in argv[1:] if c] or SUPPORTED_CITIES
    print(f"Fetching POIs for: {', '.join(cities)}", flush=True)
    print(f"Output directory:  {OUT_DIR}\n", flush=True)
    for city in cities:
        fetch_city(city)
        print()
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
