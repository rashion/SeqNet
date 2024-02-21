# What version of Python do you have?
import sys

import tensorflow.keras
import tensorflow as tf


print()
print(f"Python {sys.version}")
print(f"Tensor Flow Version: {tf.__version__}")
# print(f"Keras Version: {tensorflow.keras.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
