#!/usr/bin/env bash
# Starts the three processes of the bot inside one container.
set -euo pipefail

MODEL=models/20261004-141855-sparse-octagon.tar.gz

# Action server: endpoints.yml points Rasa at localhost:5055.
python -m rasa_sdk --actions actions --port 5055 &

# Rasa: REST channel only (no --enable-api), bound to the loopback interface.
rasa run --endpoints endpoints.yml --model "$MODEL" -i 127.0.0.1 -p 5005 &

# Wait until Rasa has loaded the model, so the first message does not fail.
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:5005/ > /dev/null && break
  sleep 2
done

# The only public process; RASA_URL defaults to localhost:5005 in the app.
exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=7860
