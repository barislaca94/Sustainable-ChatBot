"""Run whole conversations through a trained model and fail on a hang.

None of the other tests could have caught the bug this exists for. The NLU
tests only classify messages; `rasa test core` replays stories and never runs
the prediction loop against a live tracker; the unit tests stub everything.
Meanwhile three ordinary messages — "hey", "How are you", "What is 2+2" —
put the server into an endless loop inside Rasa's own dialogue manager and
the chat UI reported "Could not reach the bot".

This script drives the same code path the server uses (`Agent.handle_text`),
turn by turn, with the real action server, and checks every turn for:
  - taking longer than TURN_LIMIT seconds (a hang, or close to one)
  - producing no reply at all (a silent failure)
  - not containing the expected text, where one is given (wrong behaviour —
    the first version checked only speed, and reported "ok" while the form
    accepted "What is 2+2" as a destination)

If a turn exceeds HARD_LIMIT the Python stack of every thread is dumped to
stderr and the script exits non-zero, so a hang points at its own cause.

Usage (the action server must be running on :5055):
    python scripts/dialogue_smoke.py [models/<model>.tar.gz]
"""
from __future__ import annotations

import asyncio
import faulthandler
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def quiet_rasa_logs() -> None:
    """Rasa logs policy decisions through structlog at DEBUG, which floods the
    output with every tracker state and bypasses `logging.disable`."""
    try:
        from rasa.utils.log_utils import configure_structlog
        configure_structlog(logging.WARNING)
    except ImportError:
        import structlog
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
        )


REPO_ROOT = Path(__file__).resolve().parent.parent

TURN_LIMIT = 3.0     # seconds; the brief: "under three seconds for critical interactions"
HARD_LIMIT = 30.0    # seconds; beyond this, dump stacks and abort

Turn = Tuple[str, Optional[str]]   # (user message, text the reply must contain)


def probe_example(intent: str) -> str:
    """First example of `intent` in the dev probe set (tests/offtopic_probe.yml).

    The safety classes are exercised with a sentence taken from the probe set
    rather than one written here, so the smoke test does not add a new
    hand-written phrasing that could drift towards any evaluation set.
    """
    current = None
    for line in (REPO_ROOT / "tests" / "offtopic_probe.yml").read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- intent:"):
            current = stripped.split(":", 1)[1].strip()
        elif current == intent and stripped.startswith("- "):
            return stripped[2:]
    raise KeyError(f"no {intent} example in tests/offtopic_probe.yml")

# Each conversation runs with a fresh sender id. Keep the ones that broke
# something in the past, with a note saying what.
#
# Expected text is matched against all reply messages joined together. Where
# a turn's answer legitimately depends on model noise, leave it as None and
# rely on the speed / empty-reply checks.
CONVERSATIONS: List[Tuple[str, List[Turn]]] = [
    ("regression: core fallback loop froze the server", [
        ("hey", "Hello!"),
        ("How are you", "Hello!"),
        ("What is 2+2", "eco-travel assistant"),
    ]),
    ("regression: bare answer outside the form got the greeting", [
        ("Barcelona", "not sure what to do with that"),
        ("1500", "not sure what to do with that"),
    ]),
    ("off-topic", [
        ("tell me a joke", "eco-travel assistant"),
        ("who wrote Hamlet", "eco-travel assistant"),
    ]),
    ("small talk", [
        ("hello", "Hello!"),
        ("who are you", "I'm a bot"),
        ("thanks", "You're welcome"),
        ("bye", "Goodbye"),
    ]),
    # /nlu_fallback forces the second failure, so the path to the handover
    # does not depend on how the model happens to score a piece of gibberish.
    ("two-stage fallback all the way to handover", [
        ("asdf qwerty zzz", "not sure what you meant"),
        ("/out_of_scope", "still didn't follow"),
        ("/nlu_fallback", "not sure what you meant"),
        ("/out_of_scope", "Ticket"),
    ]),
    ("trip form with adaptive questions", [
        ("I want to plan a sustainable trip", "where would you like to travel"),
        ("Barcelona", "travelling from"),
        ("London", "When are you planning"),
        ("next week", "budget"),
        ("400", "How important is sustainability"),
        ("high", "how do you want to get around"),
        ("train or bus only", "how long a trip"),
        ("weekend", "Trip plan"),
    ]),
    ("task requests", [
        ("eco hotels in Berlin", "Hotels in Berlin"),
        ("green transport from London to Paris", "London → Paris"),
        ("carbon footprint of a flight from Madrid to Rome", "kg CO2e"),
        ("things to do in Kyoto", "Kyoto"),
    ]),
    # Adim 2. The payload turn checks the rule and response deterministically;
    # the probe sentence checks the whole path (NLU, SafetyGate, rule).
    ("safety: booking requests", [
        ("/ask_booking", "can't make bookings"),
        (probe_example("ask_booking"), "can't make bookings"),
    ]),
    ("safety: visa, health and safety questions", [
        ("/ask_regulated_advice", "official sources"),
        (probe_example("ask_regulated_advice"), "official sources"),
    ]),
    ("safety: insults", [
        ("/insult", "hasn't been helpful"),
        (probe_example("insult"), "hasn't been helpful"),
    ]),
    ("safety: privacy questions", [
        ("/ask_privacy", "only for this conversation"),
        (probe_example("ask_privacy"), "only for this conversation"),
    ]),
    ("input validation: empty and over-long messages", [
        ("   ", "500 characters"),
        ("x" * 600, "500 characters"),
    ]),
    ("regression: form took 'What is 2+2' as the destination", [
        ("I want to plan a sustainable trip", "where would you like to travel"),
        ("What is 2+2", "Where would you like to travel?"),
        ("Lisbon", "travelling from"),
    ]),
]


async def run(model: str) -> int:
    from rasa.core.agent import Agent
    from rasa.core.utils import AvailableEndpoints

    quiet_rasa_logs()
    endpoints = AvailableEndpoints.read_endpoints(str(REPO_ROOT / "endpoints.yml"))
    agent = Agent.load(model, action_endpoint=endpoints.action)
    quiet_rasa_logs()  # loading a model can reset the configuration

    failures = 0
    for title, turns in CONVERSATIONS:
        sender = f"smoke-{int(time.time() * 1000)}"
        print(f"\n== {title}")
        for text, expected in turns:
            faulthandler.dump_traceback_later(HARD_LIMIT, exit=True)
            started = time.time()
            replies = await agent.handle_text(text, sender_id=sender)
            elapsed = time.time() - started
            faulthandler.cancel_dump_traceback_later()

            first = next((r.get("text") for r in replies if r.get("text")), None)
            everything = "\n".join(r.get("text") or "" for r in replies)
            problems = []
            if elapsed > TURN_LIMIT:
                problems.append(f"slow: {elapsed:.1f}s")
            if not replies:
                problems.append("no reply")
            elif expected and expected.lower() not in everything.lower():
                problems.append(f"expected {expected!r}")
            status = "FAIL" if problems else " ok "
            if problems:
                failures += 1
            preview = (first or "").replace("\n", " / ")[:70]
            print(f"  [{status}] {elapsed:4.1f}s  {text!r:40} -> {preview}"
                  + (f"   <-- {', '.join(problems)}" if problems else ""))

    print(f"\n{failures} failing turn(s)")
    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        models = sorted((REPO_ROOT / "models").glob("*.tar.gz"))
        if not models:
            print("No trained model found. Run `rasa train` first.")
            return 1
        model = str(models[-1])
    print(f"model: {model}")
    return asyncio.run(run(model))


if __name__ == "__main__":
    sys.exit(main())
