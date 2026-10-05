# Single-container image for a Hugging Face Docker Space.
# Rasa, the action server and Streamlit run in one container; only Streamlit
# is exposed (port 7860, the Space's app_port). Rasa listens on 127.0.0.1; the
# action server listens inside the container (rasa_sdk's default host), but
# only the app port is reachable from outside the Space.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000 (Hugging Face Docker Spaces docs).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    RASA_TELEMETRY_ENABLED=false
WORKDIR /home/user/app

COPY --chown=user requirements-rasa.txt requirements-actions.txt requirements-streamlit.txt ./
RUN pip install --user --upgrade "pip<25" \
    && pip install --user -r requirements-rasa.txt -r requirements-actions.txt -r requirements-streamlit.txt

COPY --chown=user config.yml domain.yml endpoints.yml credentials.yml streamlit_app.py ./
# Turns off Streamlit's usage statistics.
COPY --chown=user .streamlit/ .streamlit/
COPY --chown=user data/ data/
COPY --chown=user components/ components/
COPY --chown=user actions/ actions/
COPY --chown=user models/20261004-141855-sparse-octagon.tar.gz models/
COPY --chown=user start.sh ./

EXPOSE 7860
CMD ["bash", "start.sh"]
