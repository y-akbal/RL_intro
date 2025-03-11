import jax 
from jax import numpy as jnp
from jax import random
import flax
from flax import linen as nn
from typing import List, Any, Callable
import numpy as np
from typing import Union
from functools import partial
class QNetwork(nn.Module):
    dtype: Any
    features: List[int] = flax.struct.field(default_factory=lambda: [128, 128])
    action_dim: int = 4
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    precision:Any = jax.lax.Precision("bfloat16")
    epsilon: float = 1e-1


    def setup(self):
        self.layers = [nn.Dense(feat, dtype = self.dtype, precision = self.precision) for feat in self.features]
        self.q_values = nn.Dense(self.action_dim, dtype=self.dtype)
        self.layer_norm = [nn.LayerNorm(dtype = self.dtype) for feat in self.features]
    
    def __call__(self, x):
        for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norm)):
            x = ln(x)
            x = layer(x)
            x = self.activation(x)
        x = self.q_values(x)
        return x


"""
network = QNetwork(dtype=jnp.float32, features=[128, 128, 128], action_dim=4)
params = network.init(random.PRNGKey(42), 5*jnp.ones((1, 8)))
q = network.apply(params, jax.random.normal(random.PRNGKey(42), (5, 8)))


ln = nn.LayerNorm()
params = ln.init(random.PRNGKey(42), 5*jnp.ones((1, 4)))
ln.apply(params, q).mean(-1)"
"""