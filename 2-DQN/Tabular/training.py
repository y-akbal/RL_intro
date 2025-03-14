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
MAX_ITERS = 4096
ENV_NAME = "CartPole-v1"
ACTION_DIM = 2
STATE_DIM = 4
RENDER_MODE = None
GAMMA = 0.95
EPSILON = 0.9
### Buffer ARGS 
SEED = 42
BUFFER_SIZE = 1024
BATCH_SIZE = 32
## Model args
LEARNING_RATE = 1e-4
EPOCHS = 15120
SOFT_UPDATE_EPS = 0.1 # EMA
INT_FEATURES = [256, 256]



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
        
        state_t_2, info = env.reset()
        state_t_1, info = env.reset()
        ## EPSILON COSINE DECAY
        #EPSILON = EPSILON 

        avr_score = []
        for iter in range(MAX_ITERS):
            ## Get an action from the network
            action = env.action_space.sample()
            ## Get action
            if np.random.rand() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.asarray(forward(params, state))
                
            state_t, reward, terminated, truncated, info = env.step(action)
            buffer.push(state_t_1, action, reward, state_t, terminated+truncated)
            state_t_1 = state_t
            avr_score.append(reward)
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
                
        #if epoch % 50 == 0:
        #    param_target = jax.tree.map(lambda x: x.copy(), params)
            
        range_.set_postfix({"reward":sum(avr_score), "EPS":EPSILON})
        param_target = jax.tree.map(lambda x,y: (1- SOFT_UPDATE_EPS)*x + SOFT_UPDATE_EPS*y, param_target, params)
        if epoch % 700 == 0:
            ## Log mean to range
            #print(buffer.reward_buffer.mean())
            RENDER_MODE = "human"
        else:
            RENDER_MODE = None

        
        env.close()


if __name__ == "__main__":
    main()














