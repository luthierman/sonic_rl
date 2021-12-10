
import matplotlib.pyplot as plt
from torch.optim import *
import torch
from collections import deque
from hyper_parameters import *
import random
import numpy as np
import os
import time
import datetime
from experience_replay import *
import sonic_env
from models import Model
class DQN(object):
    def __init__(self) -> None:
        # GYM environment
        self.env = sonic_env.make_env("SonicTheHedgehog-Genesis", "GreenhillZone.Act1")
        self.action_space = self.env.action_space.n
        self.state_space = (4,84,84)

        # HYPERPARAMETERS
        self.lr = LR
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch = BATCH_SIZE
        self.episodes = N_EPISODES
        # Q-network and Target-network
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = Model((4,84,84),self.action_space)
        print("Q-NETWORK:\n", self.q_network)
        self.opt = Adam(self.q_network.parameters(), lr=self.lr)
        self.target = Model((4,84,84),self.action_space)
        self.sync_weights()
        if use_cuda:
            print("GPU being used:", torch.cuda.get_device_name(0))
            self.q_network.cuda(self.device)
            self.target.cuda(self.device)
        self.target.eval()
        self.loss_fn = torch.nn.MSELoss()

        # DQN setup
        self.buff = 2000
        self.memory = ER_Memory(self.buff)
        self.counter = 0
        self.update_target = 1
        self.step = 0
        self.train_start = 1
        self.current_episode = 0

        # stat tracking
        self.rewards = []
        self.losses = []
        self.accuracies = []
        self.q_values = []
        self.episode_times = []
        self.episode_time = 0
        self.no_steps = []
        self.avg = 0
        self.total_reward = 0
        self.avg_reward = deque(maxlen=self.episodes)
        self.rs = deque(maxlen=50)

        # save model
        self.log_path = os.getcwd() + "\logs"
        self.name = "DQN_{}".format(datetime.datetime.now())

    def preprocess_state(self, x):
        state = np.stack(x)
        state = torch.from_numpy(state).float().to(self.device)
        return state

    def run_episode(self):
        start_time = time.time()
        s1 = self.env.reset()
        steps = 0
        done = False
        self.total_reward = 0
        total_loss = 0
        while not done:
            self.env.render()
            action = self.get_action(s1)

            s2, reward, done, _ = self.env.step(action)
            self.total_reward += reward
            self.remember(s1,
                          action,
                          reward,
                          s2,
                          done)
            if done:
                self.episode_time = time.time() - start_time
                self.episode_times.append(self.episode_time)
                self.rewards.append(self.total_reward),
                self.losses.append(total_loss),
                self.rs.append(self.total_reward)
                self.no_steps.append(steps)
                self.avg = np.mean(self.rs)
                self.avg_reward.append(self.avg)
                self.current_episode += 1
                break
            s1 = s2
            total_loss += self.learn()

            steps += 1
            if steps % 10 == 0:
                self.sync_weights()

    def learn(self):
        if len(self.memory) < self.train_start:
            return 0

        minibatch = self.memory.sample(min(len(self.memory), self.batch))
        states = self.preprocess_state(minibatch[:, 0])
        actions = self.preprocess_state(minibatch[:, 1]).type(torch.int64).unsqueeze(-1)
        rewards = self.preprocess_state(minibatch[:, 2])
        next_states = self.preprocess_state(minibatch[:, 3])
        dones = self.preprocess_state(minibatch[:, 4])
        # DQN

        self.q_network.train()
        self.target.eval()
        # Q
        Q = self.q_network.forward(states).gather(1, actions).squeeze(-1)  # Q(s, a, wq)
        # target
        Q_next = self.target.forward(next_states).max(1)[0].detach()  # max _a Q(ns, a, wt)
        y = rewards + self.gamma * (1 - dones) * Q_next  # bellman
        self.opt.zero_grad()
        loss = self.loss_fn(y, Q)
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

        return loss.item()

    def sync_weights(self):
        self.target.load_state_dict(self.q_network.state_dict())

    def get_action(self, obs):
        # obs= torch.from_numpy(obs.__array__()).unsqueeze(0).float().to(self.device)

        if len(self.memory) > self.batch:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            self.q_network.eval()
            obs = self.preprocess_state([obs])
            return self.q_network(obs).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)
        self.counter += 1


torch.cuda.empty_cache()
agent_dqn = DQN()
print(agent_dqn.name)
for i in range(agent_dqn.episodes):
    agent_dqn.run_episode()
    print("\rEpisode {}/{} [{} sec.]|| Current Avg {}, Episode Reward {}, Steps {}, eps {}".format(
        agent_dqn.current_episode,
        agent_dqn.episodes,
        np.round(agent_dqn.episode_time, 3),
        agent_dqn.avg,
        agent_dqn.total_reward,
        agent_dqn.no_steps[i],
        agent_dqn.epsilon
    ), flush=True, end="")

plt.plot(np.arange(1,201), agent_dqn.rewards)
plt.plot(np.arange(1,201), agent_dqn.avg_reward)

plt.title(agent_dqn.name)
plt.show()