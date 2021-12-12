
# https://pythonprojects.io/ai-plays-sonic/
from models import Model
import retro

import gym_super_mario_bros


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# env = sonic_env.make("SonicTheHedgehog-Genesis", "GreenhillZone.Act1")
done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()