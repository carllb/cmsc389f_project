import json
import matplotlib.pyplot as plt
import numpy as np
import time
import PIL import imageio # TODO: do we need this
from IPython import display



import gym, gym.spaces, time


env = gym.make('Breakout-ram-v0')

env.reset()

done = False

while not done:
    action = env.action_space.sample()
    print (action)
    observation, reward, done, info = env.step(action)
    
    env.render()
    time.sleep(1.0/30)