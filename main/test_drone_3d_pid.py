import gym
import gym_drone
import numpy as np

env = gym.make('Drone3D-v0',mode="PID_Controller")
env.reset()

while 1:
    state = env.state

    env.step(1)
    env.render()




