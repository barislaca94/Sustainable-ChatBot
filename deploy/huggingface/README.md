---
title: EcoFriendly ChatBot
emoji: 🌖
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Eco-Travel Advisor

A Rasa 3.6 chatbot for sustainable trip planning, with a Streamlit interface.
Source code, setup instructions and documentation:
https://github.com/barislaca94/Sustainable-ChatBot

This Space runs Rasa, the custom action server and Streamlit in one container;
only the Streamlit interface is reachable. Conversations are kept in memory
while the Space runs and are not stored. A human-advisor handover writes a
context summary to the Space's log; no human advisor is connected.
