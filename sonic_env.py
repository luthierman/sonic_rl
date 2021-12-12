import gym
import numpy as np
import cv2
from gym.spaces import Box
from gym.wrappers import FrameStack
import torch
import torchvision.transforms as T
from retro import make


class PreprocessFrame(gym.ObservationWrapper):

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.w = 84
        self.h = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.h, self.w, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = frame.__array__()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[1:-1, 1:-1]
        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        # frame =
        return frame


class ActionsDiscretizer(gym.ActionWrapper):
    # look on youtube for "Build an A2C agent that learns to play Sonic with Tensorflow"
    # for actual explanation of this
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [["LEFT"],
                   ["RIGHT"],
                   ["LEFT", "DOWN"],
                   ["RIGHT", "DOWN"], ["DOWN"],
                   ["DOWN", "B"],
                   ["B"]]
        self._actions = []

        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    def reward(self, reward):
        return reward * .01


class AllowBacktracking(gym.Wrapper):
    """
    don't punish sonic for exploring backwards
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._cur_x += reward

        reward = max(0, self._cur_x - self._max_x)  # position
        self._max_x = max(self._max_x, self._cur_x)
        return observation, reward, done, info


def make_env(game, level,scale_rew=True, record=None ):
    env = make(game=game, state=level, record=record)
    env = ActionsDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = PreprocessFrame(env)
    env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    return env


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

