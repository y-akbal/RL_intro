import jax 
from jax import numpy as jnp
from jax import random
from jax import grad, jit, vmap
import numpy as np
import flax
from flax import linen as nn
global_key = random.PRNGKey(42)
from typing import List
class QNetwork(nn.Module):
    features: List[int]
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        self.dense1 = nn.Dense(20)
        self.dense2 = nn.Dense(20)
        self.dense3 = nn.Dense(1)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = nn.relu(x)
        x = self.dense3(x)
        return x

network = QNetwork(features=[128, 128])
params = network.init(global_key, jnp.ones((128,)))
