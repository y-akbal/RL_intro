import gymnasium as gym
import ale_py
import numpy as np
import jax 
from jax import numpy as jnp
from jax import random
import flax
from flax import linen as nn
import optax
from typing import List, Any, Callable
from model import QNetwork
from buffer_play import BufferPlay



### Buffer ARGS 
SEED = 42
BUFFER_SIZE = 1024
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON = 0.1
LEARNING_RATE = 1e-3
EPOCHS = 1000
## Model args
INT_FEATURES = [128, 128]
ACTION_DIM = 4
STATE_DIM = 8
## This way network 8 -> 128 -> 128 -> 4
SOFT_UPDATE_EPS = 1e-3
## 
## Global random key
glob_key = random.PRNGKey(SEED)

def main():
    qnetwork = QNetwork(dtype=jnp.bfloat16, features=INT_FEATURES, action_dim=ACTION_DIM)
    params = qnetwork.init(glob_key, jnp.ones((1, STATE_DIM)))
    optimizer = optax.adam(LEARNING_RATE).create(params)
    buffer = BufferPlay(action_dim=ACTION_DIM, state_dim=STATE_DIM, buffer_size=BUFFER_SIZE)

    

    



if __name__ == "__main__":
    main()



















gym.register_envs(ale_py)

# Initialise the environment
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()



