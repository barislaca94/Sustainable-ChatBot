# Imported whenever Rasa loads this project's custom components, so it runs
# before the model's TensorFlow components are restored (see keras_compat.py).
from components import keras_compat  # noqa: F401
