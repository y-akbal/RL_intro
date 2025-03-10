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
    epsilon: float = 1e-1


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

    def policy(self, 
                  params, 
                  state: jnp.ndarray, 
                  prng_key: jax.random.PRNGKey, 
                  epsilon:float):
        if jax.random.uniform(prng_key) < epsilon:
            return jax.random.randint(prng_key, (1,), 0, self.action_dim)
        else:
            return jnp.argmax(self.apply(params, state, train = False), axis = -1)

global_key = random.PRNGKey(42)
network = QNetwork(dtype=jnp.bfloat16, features=[128, 128], action_dim=4)
params = network.init(global_key, jnp.ones((1, 8)))
for _ in range(100):
    random_key = random.fold_in(global_key, _)
    print(network.policy(params, jnp.ones((1, 8)), random_key, 0.9))


p = params["params"]

jax.tree.map(lambda x,y : (1-0.1)*x + 0.1*y, p, p)

"""  
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

"""