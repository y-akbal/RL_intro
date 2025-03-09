import numpy as np
from typing import List, Union


class Environment:
    def __init__(self, 
                 n_people:int = 10, 
                 likes:int = 5,
                 n_items:int = 100,
                 epsilon:float = 0.1,
                 seed:int = 42,
                 ):
        np.random.seed(seed)
        self.people = np.array([np.random.choice(range(n_items), size = likes, replace = False) for _ in range(n_people)])
        self.epsilon = epsilon
    def check(self, x:np.ndarray, y:np.ndarray):
        mask = np.random.binomial(1, 1 - self.epsilon, size = y.shape) ## Sometimes we do not like the item we are recommended
        return np.sum(np.isin(x, np.where(mask, y, -1))) 
    
    def order_check(self, x:np.ndarray, y:np.ndarray):
        pass
        






    def query(self, people:List[int], recommendations:np.ndarray) -> float:
        people_under_consideration = self.people[people, :]
        return np.array([self.check(recommendations[i], people_under_consideration[i]) for i in range(len(people))])
    

    def ordered_query(self, people:List[int], recommendations:np.ndarray) -> float:
        people_under_consideration = self.people[people, :]




        

"""
env = Environment(
    n_people = 1000,
    likes = 5,
    n_items = 10,
    epsilon = 0.2
)

people = [1, 2, 3, 4, 5]
recommendations = env.people[people, :2]
env.people[people, :], recommendations


env.query(people, recommendations)
"""