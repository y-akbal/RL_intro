
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import Environment
from model import ReC, get_recommendations
import numpy as np
from matplotlib import pyplot as plt
torch.set_float32_matmul_precision('high')

## Environment
N_ITEMS = 350
N_PEOPLE = 2750
LIKES = 10
EPSILON = 0.1
SEED = 42
##
## Model Params
EMBEDDING_DIM = 256
##
## Training Params
BATCH_SIZE = 256
EPOCHS = 1000
LEARNING_RATE = 0.001
##
## Number of Recos
RECO_SIZE = 5
## 


## Enviromnet
env = Environment(
    n_people = N_PEOPLE,
    likes = LIKES,
    n_items = N_ITEMS,
    epsilon = EPSILON
)

## Model
model = ReC(
    n_items = N_ITEMS,
    n_people = N_PEOPLE,
    embedding_dim = EMBEDDING_DIM
)
model = torch.compile(model.cuda())
## Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

def main():
    ## For some number of time 
    ## Choose as subset of people, get their recommendations and query the environment
    ## 
    avg_rewards = []
    for i in range(1000000):
        people = np.random.choice(range(N_PEOPLE), size = BATCH_SIZE, replace = False)
        people = torch.tensor(people, dtype = torch.long, device = "cuda")
        reco = get_recommendations(people, model, num_recommendations = RECO_SIZE)
        rewards = env.query(people.cpu().numpy(), reco.cpu().numpy())
        avg_rewards.append(rewards.sum())
        loss = -(torch.tensor(rewards, device = people.device).reshape(-1,1)*model(people).softmax(-1).log().gather(1, reco)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {i}, Loss: {loss.item()}, Avg Reward: {np.mean(avg_rewards[-100:])}")
        if i % 1000 == 0:
            print("Rewards: ", rewards)
    plt.figure()
    plt.plot(avg_rewards)
    plt.show()

if __name__ == "__main__":
    main()









