import numpy as np

from typing import List, Any, Callable


from abc import ABC, abstractmethod

class Buffer(ABC):
    @abstractmethod
    def push(self, state:np.ndarray, action:np.ndarray, reward:np.ndarray, next_state:np.ndarray, done:bool):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size:int = 32):
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
"""
buffer = ReplayBuffer(
    action_dim=2,
    state_dim=2,
    buffer_size=12
)

for i in range(-1,10):
    buffer.push(
    state = np.array([i, i+1]),
    action = np.array([i, i+1]),
    reward = np.array([i]),
    next_state = np.array([i+1, i+2]),
    done = False
    )

    buffer.sample()["state"]

buffer.action_buffer, buffer.reward_buffer, buffer.state_buffer, buffer.terminate_buffer

buffer.sample(12)
"""