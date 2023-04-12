from Common.env import *
import gymnasium as gym
import time
import numpy as np
import keyboard

env = SuperMario(size=10, render_fps=60)#gym.make("gym_examples/GridWorld-v0")
env.reset()

while True:
    env.render()
    time.sleep(0.001)


    if keyboard.is_pressed("q+z"):
        action = 5
    elif keyboard.is_pressed("z+d"):
        action = 1
    elif keyboard.is_pressed("z"):
        action = 3
    elif keyboard.is_pressed("d"):
        action = 0
    elif keyboard.is_pressed("q"):
        action = 2
    else:
        action = 4
    
    next_state, r, d, info = env.step(action)
    if d or info["flag_get"]:
        env.reset()