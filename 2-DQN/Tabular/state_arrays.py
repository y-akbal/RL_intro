import numpy as np


class StateArrays:
    def __init__(self, 
                 state_dim:int = 32,
                 lags:int = 2,
                 dtype:np.dtype = np.float32
                 ):
        self.state_dim = state_dim
        self.lags = lags
        self.dtype = dtype
        self._state = np.zeros((state_dim*lags,), dtype = dtype)
    
    def set_state(self, last_state:np.ndarray):
        self._state[:-self.state_dim] = self._state[self.state_dim:]
        self._state[-self.state_dim:] = last_state

    def reset(self):
        self._state = np.zeros((self.state_dim*self.lags,), dtype = self.dtype)   
    
    
    @property
    def shape(self):
        return self._state.shape
    
    def __len__(self):
        return len(self._state)
    
    @property
    def state(self):
        return self._state
    

