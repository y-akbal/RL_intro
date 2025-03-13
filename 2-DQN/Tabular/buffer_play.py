import numpy as np

from typing import List, Any, Callable


from abc import ABC, abstractmethod

class Buffer(ABC):
    @abstractmethod
    def push(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

## Very simple buffer play class that stores the state, action, reward, next state and done in numpy arrays
## This is not the most efficient way to store the data but it is simple and easy to understand
class ReplayBuffer(Buffer):
    def __init__(self, 
                action_dim:int = 4,
                state_dim:int = 8,
                buffer_size:int = 1024,
                ):
        ## State buffer is 2*state_dim because we are storing the state and next state in the same buffer
        self.state_buffer = np.zeros((buffer_size, 2*state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.int64)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.terminate_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.ptr = 0
        self.full = 0
        self.temp_reward = 0



    def push(self, 
             state:np.ndarray, 
             action:np.ndarray, 
             reward:np.ndarray, 
             next_state:np.ndarray, 
             done:bool):

        self.state_buffer[self.ptr] = np.concatenate([state, next_state])
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.terminate_buffer[self.ptr] = done
        self.ptr += 1
        self.temp_reward = (0.95)*self.temp_reward + 0.05*reward
        if self.ptr == self.state_buffer.shape[0]:
            self.ptr = 0
            self.full = 1

    def sample(self, batch_size:int = 32):
        idx = np.random.randint(0, self.state_buffer.shape[0], batch_size)
        return {
            "state":self.state_buffer[idx, :self.state_buffer.shape[1]//2],
            "action":self.action_buffer[idx],
            "reward":self.reward_buffer[idx],
            "next_state":self.state_buffer[idx, self.state_buffer.shape[1]//2:],
            "done":self.terminate_buffer[idx]
        }            
    @property
    def reward(self):
        return self.temp_reward
"""
buffer = ReplayBuffer(
    action_dim=1,
    state_dim=2,
    buffer_size=12
)

from model import QNetwork

import jax
from jax import numpy as jnp

network = QNetwork(features=[128], action_dim=2, dtype=jnp.float32)

params = network.init(jax.random.PRNGKey(42), jnp.ones((1, 2)))
batch = buffer.sample()
staes, actions = batch["state"], batch["action"]

jnp.take_along_axis(network.apply(params, staes), actions, axis=1)

"""