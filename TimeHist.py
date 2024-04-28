import tensorflow as tf
from tensorflow import keras
import time
# from https://stackoverflow.com/a/43186440
class TimeHistory(keras.callbacks.Callback):
    # Initialize the times array
    def on_train_begin(self, logs={}):
        self.times = []

    # Record when a new epoch has begun
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    # Calculate how long it took for the epoch to be completed
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


