"""
File: TimeHist.py
Description: Defines the custom callback class to record the time each epoch took
Author: Alexander Lemieux
Actually created by: https://stackoverflow.com/a/43186440
Note: This class is so simple that I could have written it by looking at the documentation
but, I felt that rewriting it would be pointless
as I would just be changing variable names
"""

import tensorflow as tf
from tensorflow import keras
import time


# from https://stackoverflow.com/a/43186440
class TimeHistory(keras.callbacks.Callback):
    # Initialize TimeHist object
    def __init__(self):
        super().__init__()
        self.epoch_time_start = None
        self.times = None

    # Initialize the times array
    def on_train_begin(self, logs={}):
        self.times = []

    # Record when a new epoch has begun
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    # Calculate how long it took for the epoch to be completed
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
