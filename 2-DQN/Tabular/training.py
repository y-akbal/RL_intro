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
from buffer_play import ReplayBuffer


### Enviroment ARGS
MAX_EPISODE_LENGTH = 1000
### Buffer ARGS 
SEED = 42
BUFFER_SIZE = 1024
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON = 0.1
LEARNING_RATE = 1e-3
ITERS = 1000
## Model args
INT_FEATURES = [128, 128]
ACTION_DIM = 4
STATE_DIM = 8
## This way network 8 -> 128 -> 128 -> 4
SOFT_UPDATE_EPS = 1e-3
## 
## Global random key
glob_key = random.PRNGKey(SEED)

def grad_fn(params, target_params, state, action, reward, next_state, done):

    @jax.jit
    def loss_fn(params, state, action, reward, next_state, done):
        q_values = QNetwork.apply(params, state)
        next_q_values = QNetwork.apply(target_params, next_state)
        target = reward + GAMMA * (1.0 - done) * jnp.max(next_q_values, axis=1)
        return jnp.mean((q_values - target) ** 2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(params, state, action, reward, next_state, done)
    return loss, grad

@jax.jit
def param_soft_update(params, target_params, epsilon = SOFT_UPDATE_EPS):
    return jax.tree.map(lambda p, tp: (1.0 - epsilon) * tp + epsilon * p, params, target_params)
    
def main():
    qnetwork = QNetwork(dtype=jnp.bfloat16, features=INT_FEATURES, action_dim=ACTION_DIM)
    params = qnetwork.init(glob_key, jnp.ones((1, STATE_DIM)))
    optimizer = optax.adam(LEARNING_RATE).init(params)
    buffer = ReplayBuffer(action_dim=ACTION_DIM, state_dim=STATE_DIM, buffer_size=BUFFER_SIZE)

    for iter in range(ITERS):
        
        for _ in range(MAX_EPISODE_LENGTH):
        ## There should be one more loop here
        ## for each step in the environment
        ## you should sample from the buffer
        ## and update the network
        ## push the state, action, reward, next_state and done to the buffer
        ## and then sample from the buffer
        ## Remember that your network has batch norm so you should set train to True
        ## when you are updating the network
        ## and False when you are sampling from the buffer
        ## Also remember to update the target network gradually --> soft update

        

    



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



