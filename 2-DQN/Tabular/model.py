import jax 
from jax import numpy as jnp
from jax import random
import flax
from flax import linen as nn

from typing import List, Any, Callable

class QNetwork(nn.Module):
    dtype: Any
    features: List[int] = flax.struct.field(default_factory=lambda: [128, 128])
    action_dim: int = 4
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    precision:Any = jax.lax.Precision("bfloat16")


    def setup(self):
        self.layers = [nn.Dense(feat, dtype = self.dtype, precision = self.precision) for feat in self.features]
        self.q_values = nn.Dense(self.action_dim, dtype=self.dtype)
        self.batch_norm = [nn.BatchNorm(dtype = self.dtype) for feat in self.features]
    
    def __call__(self, x, train = False):
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norm)):
            x = bn(x, use_running_average=not train)
            x = layer(x)
            x = self.activation(x)
        x = self.q_values(x)
        return x

    
import optax
import jax
import jax.numpy as jnp
@jax.jit
def rosenbrock(x):
    return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


x = jnp.array([1.3, 0.7, 0.8, 1.9, 1.2])
value, grad = jax.value_and_grad(rosenbrock)(x)

from optax import adam

optimizer = adam(1e-3)
opt_state = optimizer.init(x)
for _ in range(1000):
    value, grad = jax.value_and_grad(rosenbrock)(x)
    updates, opt_state = optimizer.update(grad, opt_state)
    x = optax.apply_updates(x, updates)
    print(value)


    