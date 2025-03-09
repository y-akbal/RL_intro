
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import Environment
from model import ReC, get_recommendations
import numpy as np
from matplotlib import pyplot as plt
from tools import GetIndex
torch.set_float32_matmul_precision('high')

## Environment
N_ITEMS = 1000
N_PEOPLE = 5000
LIKES = 5
EPSILON = 0.1
SEED = 42
## Model Params
EMBEDDING_DIM = 256
## Training Params
BATCH_SIZE = 256
EPOCHS = 1000000
LEARNING_RATE = 0.0001
## Number of Recos
RECO_SIZE = 5
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
).cuda()
model = torch.compile(model)
## Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

def main():
    ## For some number of time 
    ## Choose as subset of people, get their recommendations and query the environment
    ## 
    avg_rewards = []
    batcher = GetIndex(N_PEOPLE, BATCH_SIZE)
    for i in range(EPOCHS):
        people = batcher.take()
        people = torch.tensor(people, dtype = torch.long, device = "cuda")
        model.eval()
        reco = get_recommendations(people, model, num_recommendations = RECO_SIZE)
        rewards = env.query(people.cpu().numpy(), reco.cpu().numpy())
        avg_rewards.append(rewards.sum())
        # Loss
        model.train()
        loss = -(torch.tensor(rewards, device = people.device).reshape(-1,1)*model(people, add_noise = True).softmax(-1).log().gather(1, reco)).mean()
        # Collect Gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {i}, Loss: {loss.item()}, Avg Reward: {np.mean(avg_rewards[-100:])}")
        if i % 1000 == 0:
            print("Rewards: ", rewards)
    plt.figure()
    plt.plot(avg_rewards)
    plt.title("Training")
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.savefig("training_no_value.png")
    plt.show()
  



if __name__ == "__main__":
    main()









