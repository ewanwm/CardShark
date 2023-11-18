import numpy as np
import PIL
from enum import Enum
import gym
import gym.spaces
from Logging import *
import tensorflow as tf
import tf_agents as Agents
import tkinter
import xml.etree
from CoupEngine import *

INFO("All modules loaded")

gym.register("Coup_Game", Game)
game = gym.make("Coup_Game", nPlayers=4)
obs, reward, terminated, truncated, info = game.reset()

for _ in range(100):
    obs, reward, terminated, truncated, info = game.step(game.action_space.sample(mask=info["mask"])) ##game.action_space.sample())


