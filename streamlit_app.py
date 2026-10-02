import os
import uuid

import streamlit as st
import requests

# Local development uses localhost; docker-compose overrides via env var.
RASA_URL = os.environ.get(
    "RASA_URL",
    "http://localhost:5005/webhooks/rest/webhook",
)


def sender_id() -> str:
    """Return this browser session's conversation id.

    Rasa keys its tracker (slots, active form, handover state) on the sender
    id. A hardcoded id would make every browser tab share one conversation,
    so each Streamlit session gets its own.
    """
    if "sender_id" not in st.session_state:
        st.session_state["sender_id"] = f"web-{uuid.uuid4().hex[:8]}"
    return st.session_state["sender_id"]


def reset_conversation() -> None:
    """Clear the local transcript AND the Rasa-side tracker.

    Clearing only st.session_state leaves slots, an active form and the
    handover flag alive on the server, so the "new" conversation would
    resume mid-form. `/restart` is Rasa's built-in intent for this.
    """
    try:
        requests.post(
            RASA_URL,
            json={"sender": sender_id(), "message": "/restart"},
            timeout=10,
        )
    except requests.RequestException:
        # The bot may be down; the local reset below still has to happen.
        pass
    st.session_state["messages"] = []
    st.session_state["handover_active"] = False
    st.session_state["sender_id"] = f"web-{uuid.uuid4().hex[:8]}"


def render_bot_message(text: str) -> None:
    """Render a bot reply with colour-coded cards or handover banner based on prefix.

    - 🎫 → red error banner (human handover)
    - 🟢 → green success card (low emission / strong eco signals)
    - 🟡 → yellow warning card (moderate)
    - 🔴 → red error card (high emission)
    - ⚠️ / ℹ️ → blue info box (disclaimers, caveats)
    - anything else → plain text

    Every band emoji is paired with a word ("Low emission", "Moderate",
    "High emission") in the action text, so the colour is never the only
    carrier of meaning — screen readers and colour-blind users get the same
    information.
    """
    stripped = text.lstrip()
    if stripped.startswith("🎫"):
        st.error(text)
    elif stripped.startswith("🟢"):
        st.success(text)
    elif stripped.startswith("🟡"):
        st.warning(text)
    elif stripped.startswith("🔴"):
        st.error(text)
    elif stripped.startswith("⚠️") or stripped.startswith("ℹ️"):
        st.info(text)
    else:
        st.write(text)


def call_rasa(message: str, sender: str) -> list[dict]:
    """Send a message to the Rasa REST endpoint and return the full reply payload.

    Each reply entry is a dict with at least a "text" key and optionally a
    "buttons" key (list of {"title", "payload"} dicts sent by the bot).
    """
    try:
        response = requests.post(
            RASA_URL,
            json={"sender": sender, "message": message},
            timeout=30,
        )
        payload = response.json()
    except requests.RequestException:
        return [{"text": "Could not reach the bot. Is Rasa running?"}]

    if not payload:
        return [{"text": "I did not understand that."}]

    return payload


# ---------- Sidebar ----------
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this panel to configure the bot")
page = st.sidebar.selectbox("Go to", ["Home", "Chat", "About"])

if st.sidebar.button("Clear Chat"):
    reset_conversation()

message_count = len(st.session_state.get("messages", []))
st.sidebar.write(f"Messages sent: {message_count}")
st.sidebar.caption(f"Conversation id: `{sender_id()}`")

if st.session_state.get("handover_active"):
    st.sidebar.error("🎫 Handover active — a human advisor has your context.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Privacy: your messages are kept only for this session. No persistent storage. GDPR-compliant."
)

# ---------- Main area ----------
st.title("Eco-Travel Advisor")
st.divider()

if page == "Home":
    st.header("Welcome")
    st.write(
        "This bot helps you plan sustainable trips, look up carbon footprints, "
        "check the weather and convert currencies."
    )
    st.markdown(
        """
        **What you can try:**
        - Plan a sustainable trip (multi-turn adaptive form with validation)
        - Eco-friendly hotels in a city
        - Green transport between two cities
        - Carbon footprint of a flight / train / car / bus
        - Suggested local, community-friendly activities
        - Carbon offset programs
        - Human handover for complex or ambiguous requests
        """
    )
    st.info("Go to the **Chat** page from the sidebar to start.")

elif page == "Chat":
    st.header("Chat")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "handover_active" not in st.session_state:
        st.session_state["handover_active"] = False

    if st.session_state["handover_active"]:
        st.error("🎫 A human travel advisor has taken over this conversation. You can still chat with the bot.")

    # ---- Display history ----
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_bot_message(message["content"])
            else:
                st.write(message["content"])

    # ---- Bot-sent buttons: only render for the LAST assistant reply ----
    button_input = None
    assistant_msgs = [m for m in st.session_state["messages"] if m.get("role") == "assistant"]
    if assistant_msgs and assistant_msgs[-1].get("buttons"):
        last_buttons = assistant_msgs[-1]["buttons"]
        cols = st.columns(min(len(last_buttons), 3))
        for i, b in enumerate(last_buttons):
            with cols[i % 3]:
                # Include the message count in the key so re-shown buttons stay unique.
                key = f"botbtn_{len(st.session_state['messages'])}_{i}"
                if st.button(b["title"], key=key):
                    # Store the human-readable title so the transcript shows the
                    # click as if the user had typed it.
                    button_input = (b["payload"], b["title"])

    # ---- Hard-coded shortcut buttons ----
    quick_input = None

    st.write("**Weather & currency:**")
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        if st.button("Weather in Berlin"):
            quick_input = "what is the weather in Berlin?"
    with row3_col2:
        if st.button("Weather in Istanbul"):
            quick_input = "what is the weather in Istanbul?"
    with row3_col3:
        if st.button("USD to EUR"):
            quick_input = "exchange rate from USD to EUR"

    st.write("**Eco-travel:**")
    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        if st.button("🌱 Plan a sustainable trip"):
            quick_input = "I want to plan a sustainable trip"
    with row4_col2:
        if st.button("🏨 Eco-hotels in Barcelona"):
            quick_input = "eco-friendly hotels in Barcelona"
    with row4_col3:
        if st.button("🚆 Green transport LON → PAR"):
            quick_input = "green transport from London to Paris"

    row5_col1, row5_col2, row5_col3 = st.columns(3)
    with row5_col1:
        if st.button("✈️ CO2: Madrid → Rome"):
            quick_input = "carbon footprint of a flight from Madrid to Rome"
    with row5_col2:
        if st.button("🎋 Kyoto activities"):
            quick_input = "suggest local eco activities in Kyoto"
    with row5_col3:
        if st.button("🎫 Talk to human"):
            quick_input = "I want to talk to a human"

    # ---- Chat input ----
    typed_input = st.chat_input("Type your message here...")

    # Priority: bot-button click > hard-coded shortcut > typed message
    if button_input is not None:
        # button_input is (payload, title). Send payload to Rasa but display title.
        payload_to_send, display_text = button_input
        user_input = payload_to_send
    elif quick_input:
        user_input = quick_input
        display_text = quick_input
    elif typed_input:
        user_input = typed_input
        display_text = typed_input
    else:
        user_input = None
        display_text = None

    if user_input:
        # Render + store user turn
        st.session_state["messages"].append({"role": "user", "content": display_text})
        with st.chat_message("user"):
            st.write(display_text)

        # Get bot replies (list of {"text", "buttons"?})
        replies = call_rasa(user_input, sender=sender_id())

        with st.chat_message("assistant"):
            for reply in replies:
                text = reply.get("text", "")
                buttons = reply.get("buttons") or []
                if text:
                    render_bot_message(text)
                    # Store text and buttons together so buttons can be
                    # re-rendered next tick for the latest bot reply only.
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": text,
                        "buttons": buttons,
                    })
                    if text.lstrip().startswith("🎫"):
                        st.session_state["handover_active"] = True

        # A button click already triggered a rerun via Streamlit; typed input
        # also triggers one implicitly. No manual st.rerun() needed here — that
        # avoids the recursion problem we hit earlier.

elif page == "About":
    st.header("About")
    st.write(
        "Eco-Travel Advisor built on Rasa 3.6.21 + Streamlit. "
        "A sustainable-tourism planning assistant."
    )
    st.markdown(
        """
        **Architecture:**
        - Rasa NLU (WhitespaceTokenizer + CountVectors featurisers →
          DIETClassifier + FallbackClassifier)
        - Rasa Core (RulePolicy + TEDPolicy + UnexpecTEDIntentPolicy)
        - Custom actions server (hotel ranking, activities, carbon calculators)
        - Live APIs: Nominatim (geocoding), Open-Meteo (weather),
          Frankfurter (currency), Wikipedia (place summaries)
        - OpenStreetMap POI data pre-fetched offline via Overpass

        **Sustainability features:**
        - Hotels ranked by *proxy* signals from OpenStreetMap (transit
          proximity, scale, parking). The bot never claims a property is
          eco-certified — OSM holds no reliable certification data — and says
          so in every hotel answer.
        - Colour-coded carbon bands (green / amber / red) with a word label
          alongside the colour
        - Adaptive multi-turn trip planning form (destination, dates, budget,
          sustainability level; extra questions appear based on prior answers)
        - Bot-sent quick-reply buttons (form questions and follow-ups)
        - Community-supported local activity suggestions
        - Verified carbon offset programs (Gold Standard, Atmosfair, Klima, myclimate)
        - Two-stage clarification fallback + human handover with full context
        """
    )

    st.markdown("**Data privacy:** No conversation history is persisted server-side. GDPR-compliant.")
