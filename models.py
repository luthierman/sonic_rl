import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import copy
from itertools import count
from collections import namedtuple, deque
import random
import torch.autograd as autograd
import datetime


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_shape = input_dim
        self.num_actions = output_dim
        if self.input_shape[1] != 84:
            raise ValueError(f"Expecting input height: 84, got: {self.input_shape[1]}")
        if self.input_shape[2] != 84:
            raise ValueError(f"Expecting input width: 84, got: {self.input_shape[2]}")
        self.net = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        # self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, x):

        x = self.net(x)
        x = x.view(x.size(0), -1)
        actions = self.fc(x)
        return actions

    def feature_size(self):
        return self.net(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class Model_R(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_shape = input_dim
        self.num_actions = output_dim
        if self.input_shape[1] != 84:
            raise ValueError(f"Expecting input height: 84, got: {self.input_shape[1]}")
        if self.input_shape[2] != 84:
            raise ValueError(f"Expecting input width: 84, got: {self.input_shape[2]}")
        self.net = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.lstm = nn.LSTM(64, 64, batch_first=True)
        # self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, x):

        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.lstm(x)
        actions = self.fc2(x)
        return actions

    def feature_size(self):
        return self.net(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
