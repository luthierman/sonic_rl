
# https://pythonprojects.io/ai-plays-sonic/
from models import Model
import sonic_env
env = sonic_env.make("SonicTheHedgehog-Genesis", "GreenhillZone.Act1")
done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()