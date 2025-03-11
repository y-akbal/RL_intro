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
import gymnasium as gym
from tqdm import tqdm
from functools import partial
from typing import Dict


## Jax will work cpu
jax.config.update("jax_platform_name", "cpu")

## ENV ARGS
MAX_ITERS = 256
ENV_NAME = "MountainCar-v0"
RENDER_MODE = None
### Buffer ARGS 
SEED = 42
BUFFER_SIZE = 2048
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON = 0.9
LEARNING_RATE = 1e-3
EPOCHS = 15120
SOFT_UPDATE_EPS = 1e-1 # EMA
## Model args
INT_FEATURES = [128, 128]
ACTION_DIM = 3
STATE_DIM = 2


def main():
    # Init network,
    glob_key = random.PRNGKey(SEED)
    qnetwork = QNetwork(dtype=jnp.float32, features=INT_FEATURES, action_dim=ACTION_DIM)
    params = qnetwork.init(glob_key, jnp.ones((1, STATE_DIM)))
    param_target = jax.tree.map(lambda x: x.copy(), params)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    buffer = ReplayBuffer(action_dim=1, state_dim=STATE_DIM, buffer_size=BUFFER_SIZE)

    @jax.jit
    def train_step(params, old_params, batch:Dict[str, jnp.array], GAMMA: float = GAMMA):

        def loss_fn(params):
            # Get data
            state, action, reward, next_state, done = batch["state"], batch["action"], batch["reward"], batch["next_state"], batch["done"]
            q_pred = jnp.take_along_axis(qnetwork.apply(params, state), action, axis =1).squeeze()
            target = reward.squeeze() + GAMMA*(1-done.squeeze())*jax.lax.stop_gradient(jnp.max(qnetwork.apply(old_params, next_state), axis = 1))
            loss = optax.l2_loss(q_pred, target).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)    
        return grads, loss
    
    @jax.jit
    def forward(params, x): 
        return jnp.argmax(qnetwork.apply(params, x), axis=-1)

    @jax.jit
    def update_grad(params_grad, opt_state, params):
        updates, opt_state = optimizer.update(params_grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    

    RENDER_MODE = None
    range_ = tqdm(range(EPOCHS))
    for epoch in range_:   

        sub_key = random.fold_in(glob_key, epoch)
        
        env = gym.make(ENV_NAME, render_mode=RENDER_MODE, max_episode_steps=MAX_ITERS)
        
        state, info = env.reset()
        ## EPSILON COSINE DECAY
        EPSILON = 0.1 + 0.9*(1 + np.cos(np.pi*epoch/EPOCHS))/2
        for iter in range(MAX_ITERS):
            ## Get an action from the network
            action = env.action_space.sample()
            ## Get action
            if np.random.rand() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.asarray(forward(params, state))
                
            next_state, reward, terminated, truncated, info = env.step(action)

            buffer.push(state, action, reward, next_state, terminated+truncated)
            state = next_state

            ## Do the training
            if buffer.full:
                data = buffer.sample(BATCH_SIZE)
                data_jax = {k:jnp.array(v) for k,v in data.items()}
                ## Calculate the target
                params_grad, loss = train_step(params, param_target, data_jax)
                ## Update the network
                params, opt_state = update_grad(params_grad, opt_state, params)
            ## If the episode has ended then we can reset to start a new episode
            if terminated or truncated:
                state, info = env.reset()
                break
                
        
        param_target = jax.tree.map(lambda x,y: (1- SOFT_UPDATE_EPS)*x + SOFT_UPDATE_EPS*y, params, param_target)
        if epoch % 100 == 0:
            ## Log mean to range
            range_.set_postfix({"reward":buffer.reward_buffer.mean()})
            #print(buffer.reward_buffer.mean())
            #RENDER_MODE = "human"
        else:
            RENDER_MODE = None

        
        env.close()


if __name__ == "__main__":
    main()














