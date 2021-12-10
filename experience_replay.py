from collections import namedtuple, deque
import random
import numpy as np
import pandas as pd
import torch


class ER_Memory(object):
    def __init__(self, length):
        self.memory = deque(maxlen=length)

    def remember(self, *args):
        self.memory.append([*args])

    def sample(self, batch_size):
        minibatch = np.stack(random.sample(self.memory, batch_size))
        return minibatch

    def __len__(self):
        return len(self.memory)

    def show(self):
        return pd.DataFrame(self.memory)