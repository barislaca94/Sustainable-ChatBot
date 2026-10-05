"""Make the served model load with the optimizer it was trained with, on every machine.

Why it exists
-------------
The served model was trained on an Apple-silicon Mac. There, Keras 2.12
replaces the v2.11+ Adam optimizer with the legacy one (`is_arm_mac()` and
`get()` in keras/optimizers/__init__.py), so the DIETClassifier, TEDPolicy and
UnexpecTEDIntentPolicy checkpoints hold legacy optimizer state. On Linux,
Windows and Intel Macs Keras keeps the v2.11+ Adam, which refuses that state
("You are trying to restore a checkpoint from a legacy Keras optimizer into a
v2.11+ Optimizer", keras/optimizers/optimizer.py). Rasa catches the error and
carries on with an untrained component (`DIETClassifier.load`,
`TEDPolicy.load`), so every message would end in the fallback.

How it works
------------
Rasa creates these optimizers as `tf.keras.optimizers.Adam(...)`, so pointing
that name at the legacy class gives every machine the setup used in training.
The optimizer only updates weights during training and plays no part in a
prediction. Rasa imports this package (for the SafetyGate in config.yml) while
it reads a model's graph, before any component is loaded, so the fix applies
to `rasa run`, `rasa train`, `rasa test` and the scripts alike.
"""
import tensorflow as tf

tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam
