import gym
import time
import numpy as np
env = gym.make("FrozenLake-v1",render_mode="human")
env.reset()
env.observation_space

env.render()
time.sleep(10)

# We re-initialize the Q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))

numberOfIterations=30

for i in range(numberOfIterations):
    randomAction= env.action_space.sample()
    returnValue=env.step(randomAction)
    env.render()
    print('Iteration: {} and action {}'.format(i+1,randomAction))
    time.sleep(1)
    if returnValue[2]:
        break

env.close()