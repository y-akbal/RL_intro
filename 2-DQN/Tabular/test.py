import jax 
from jax import numpy as jnp
from jax import random
import flax
import optax
from flax import linen as nn

from typing import List, Any, Callable

class a(nn.Module):
    @nn.compact
    def __call__(self, x, train = False):
        x = nn.Dense(20, use_bias=True)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(20, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(1, use_bias=True)(x)
        return x

params = a().init(random.PRNGKey(42), jnp.ones((1, 4)))

x = jax.random.normal(random.PRNGKey(42), (10000, 4))
y = 5*jax.random.normal(random.PRNGKey(42), (10000, 1))

output = a().apply(params, x)
print(params, output)

def loss_fn(params, x, y):
    pred = a().apply(params, x)
    return jnp.mean((pred - y)**2)

param_grad = jax.grad(loss_fn)(params, x, y)

optimizer = optax.sgd(1e-3, momentum = 0.95, nesterov=True)
opt_state = optimizer.init(params)



for i in range(10000):
    param_grad = jax.grad(loss_fn)(params, x, y)
    val, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    print(val)



params