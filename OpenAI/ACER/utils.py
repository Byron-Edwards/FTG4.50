# -*- coding: utf-8 -*-
import torch
from torch import multiprocessing as mp
import os
import signal, functools


# Global counter
class Counter():
    def __init__(self):
        self.val = mp.Value('i', 0)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def set(self, value):
        with self.lock:
            self.val.value = value

    def value(self):
        with self.lock:
            return self.val.value


# Converts a state from the OpenAI Gym (a numpy array) to a batch tensor
def state_to_tensor(state):
    return torch.from_numpy(state).float().unsqueeze(0)


# elif isinstance(state, list):
#   return torch.from_numpy(state[0]).float().unsqueeze(0)
# else:
#   raise TypeError


class TimeoutError(Exception): pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

