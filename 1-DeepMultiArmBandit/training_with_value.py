
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import Environment
from model import ReC, get_recommendations, Value_Network
import numpy as np
from matplotlib import pyplot as plt
from tools import GetIndex
torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
## Environment

## Environment
N_ITEMS =  1500
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
).to(device)
value_network = Value_Network(
    n_items = N_ITEMS,
    n_people = N_PEOPLE,
    embedding_dim = EMBEDDING_DIM
).to(device)
if device == "cuda" or device == "cpu":
    model = torch.compile(model.to(device))
    value_network = torch.compile(value_network.to(device))


## Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay=0.01)
value_optimizer = torch.optim.AdamW(value_network.parameters(), lr = 0.001)

def main():
    ## For some number of time 
    ## Choose as subset of people, get their recommendations and query the environment
    ## 
    avg_rewards = []
    temp_rewards = 0.0
    batcher = GetIndex(N_PEOPLE, BATCH_SIZE)
    for i in range(1000000):
        people = batcher.take()
        people = torch.tensor(people, dtype = torch.long, device = device)
        reco = get_recommendations(people, model, num_recommendations = RECO_SIZE)
        rewards = env.query(people.cpu().numpy(), reco.cpu().numpy())
        avg_rewards.append(rewards.sum())
        
        ## Value Network
        value_optimizer.zero_grad()
        value_loss = ((torch.tensor(rewards, device = device).reshape(-1,1) - value_network(people)).pow(2)).mean()
        value_loss.backward()
        value_optimizer.step()
    
        with torch.no_grad(): 
            rewards_ = torch.tensor(rewards, device = device).reshape(-1,1) - value_network(people)
            rewards_ = torch.clamp(rewards_, -0.5, 0.5)
        
        optimizer.zero_grad()
        preds = model(people).clamp(-3, 3)
        loss = -((rewards_)*preds.softmax(-1).log().gather(1, reco)).mean()
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









