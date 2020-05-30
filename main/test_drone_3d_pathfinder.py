import json

import gym
import gym_drone
import numpy as np

env = gym.make('Drone3D-v0')
env.reset()

for i, a in enumerate([15,15,15,19, 5, 2, 4, 1, 4, 4, 4, 15, 1,1, 4, 1, 1, 1, 15, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1,
             4, 4, 1, 4, 6, 6,1,0]):
    state = env.state

    env.step(a)
    env.render()
env.save_path()





