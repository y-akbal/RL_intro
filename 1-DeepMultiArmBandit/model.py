import torch
import torch.nn as nn
import torch.nn.functional as F

## Simple recommender system with a single embedding layer

class ReC(nn.Module):
    def __init__(self, 
                 n_items:int = 100,
                 n_people:int = 10,
                 embedding_dim:int = 10,
                 
                 ):
        super(ReC, self).__init__()
        self.people = nn.Embedding(n_people, embedding_dim)
        self.Network = nn.Sequential(*[
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512, ),
            nn.Linear(512, 768),
            nn.GELU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, n_items)
        ])
    def forward(self, 
                people:torch.Tensor,
                add_noise:bool = False,
                ) -> torch.Tensor:
        embeddings = self.people(people)
        if add_noise:
            embeddings += 0.02*torch.randn_like(embeddings)
        embeddings = F.normalize(embeddings, p = 2, dim = -1)
        return self.Network(embeddings)


class Value_Network(nn.Module):
    def __init__(self, 
                 n_items:int = 100,
                 n_people:int = 10,
                 embedding_dim:int = 10,
                 ):
        super().__init__()
        self.people = nn.Embedding(n_people, embedding_dim)
        self.Network = nn.Sequential(*[
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        ])
    def forward(self, 
                people:torch.Tensor,
                ) -> torch.Tensor:
        embeddings = self.people(people)
        return self.Network(embeddings)


@torch.no_grad()
def get_recommendations(people:torch.Tensor, 
                        model:ReC,
                        num_recommendations:int = 5
                        ) -> torch.Tensor:
    if isinstance(people, list):
        people = torch.tensor(people, dtype = torch.long, device = people.device).unsqueeze(0)
    soft_maxed_reco = model(people).softmax(dim = -1)
    return torch.multinomial(soft_maxed_reco, num_recommendations, replacement = False)






"""
model = ReC(
    n_items = 10,
    n_people = 50,
    embedding_dim = 32
)


people = torch.tensor([1, 2, 3, 4, 5])
people
reco = get_recommendations(people, model, num_recommendations = 3)
reco

from environment import Environment

env = Environment(
    n_people = 50,
    likes = 5,
    n_items = 10,
    epsilon = 0.2
)

rewards = env.query(people, reco)
loss = -(torch.tensor(rewards).reshape(-1,1)*model(people).softmax(-1).log().gather(1, reco)).mean()
loss.backward()
loss.item()
"""