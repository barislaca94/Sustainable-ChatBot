"""Verify the Climatiq key and the activity IDs used by actions/actions.py.

The activity IDs in `CLIMATIQ_ACTIVITY_IDS` are version-specific: Climatiq
retires and renames them between data versions, so they must be checked
against a live key rather than trusted. Run this once after putting
CLIMATIQ_API_KEY in `.env`:

    python scripts/climatiq_check.py

For every transport mode it estimates 100 passenger-km and prints the result
next to the local DEFRA figure. If an ID no longer resolves, it queries
Climatiq's /search endpoint and prints candidate replacements to paste into
`CLIMATIQ_ACTIVITY_IDS`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions.actions import (  # noqa: E402  (path set up above)
    CLIMATIQ_ACTIVITY_IDS,
    CLIMATIQ_DATA_VERSION,
    CLIMATIQ_SEARCH_URL,
    EMISSION_FACTORS,
    climatiq_activity_id,
    climatiq_api_key,
)

ESTIMATE_URL = "https://api.climatiq.io/data/v1/estimate"

# (label, mode, distance). Flights are checked at both distances because the
# activity id switches at CLIMATIQ_SHORT_HAUL_KM.
CHECKS = [(mode, mode, 100.0) for mode in CLIMATIQ_ACTIVITY_IDS] + [
    ("flight (short)", "flight", 100.0),
    ("flight (long)", "flight", 8000.0),
]

# What to search for when an activity id fails to resolve.
SEARCH_TERMS = {
    "train":  "passenger train rail",
    "bus":    "passenger bus",
    "coach":  "passenger coach",
    "car":    "passenger car average",
    "flight": "passenger flight economy",
}


def estimate(key: str, activity_id: str,
             distance_km: float) -> tuple[float | None, str]:
    """Estimate `distance_km` for one activity id. Returns (kg, message)."""
    try:
        response = requests.post(
            ESTIMATE_URL,
            json={
                "emission_factor": {
                    "activity_id": activity_id,
                    "data_version": CLIMATIQ_DATA_VERSION,
                },
                "parameters": {
                    "passengers": 1,
                    "distance": distance_km,
                    "distance_unit": "km",
                },
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        return None, f"request failed: {exc}"

    try:
        body = response.json()
    except ValueError:
        return None, f"HTTP {response.status_code}, non-JSON body"

    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {body.get('message', body)}"

    co2e = body.get("co2e")
    unit = body.get("co2e_unit", "kg")
    if co2e is None:
        return None, f"no co2e in answer: {body}"
    kg = float(co2e) * (1000.0 if unit == "t" else 1.0)
    return kg, f"{body.get('emission_factor', {}).get('source', '?')}"


def search(key: str, query: str) -> None:
    """Print candidate activity ids for a search term."""
    try:
        response = requests.get(
            CLIMATIQ_SEARCH_URL,
            params={
                "query": query,
                "data_version": CLIMATIQ_DATA_VERSION,
                "results_per_page": 5,
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"    search failed: {exc}")
        return

    for item in body.get("results", [])[:5]:
        print(
            f"    candidate: {item.get('activity_id')}\n"
            f"      source={item.get('source')} region={item.get('region')} "
            f"unit_type={item.get('unit_type')}"
        )


def main() -> int:
    key = climatiq_api_key()
    if not key:
        print(
            "CLIMATIQ_API_KEY is not set.\n"
            "Put it in .env at the repo root (see .env.example), then re-run.\n"
            "Without it the bot uses the local DEFRA table and says so."
        )
        return 1

    print(f"Data version: {CLIMATIQ_DATA_VERSION}\n")
    print(f"{'mode':16s} {'km':>6s} {'Climatiq':>10s} {'local':>9s}   source")

    failures = 0
    for label, mode, distance in CHECKS:
        activity_id = climatiq_activity_id(mode, distance)
        local_kg = distance * EMISSION_FACTORS.get(mode, 0.0)
        kg, note = estimate(key, activity_id, distance)
        if kg is None:
            failures += 1
            print(f"[FAIL] {label:16s} {activity_id}\n    {note}")
            search(key, SEARCH_TERMS.get(mode, mode))
        else:
            print(
                f"[ OK ] {label:16s} {distance:6.0f} {kg:10.2f} {local_kg:9.2f}   {note}"
            )

    print()
    if failures:
        print(
            f"{failures} activity id(s) need replacing in "
            "actions/actions.py → CLIMATIQ_ACTIVITY_IDS."
        )
        return 1
    print("All activity ids resolve. The bot will report Climatiq as its source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
