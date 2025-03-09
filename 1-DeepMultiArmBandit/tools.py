import numpy as np


class GetIndex:
    def __init__(self, 
                 n_people:int, 
                 batch_size:int,
                 shuffle:bool = True
                 ):
        self.n_people = n_people
        self.batch_size = batch_size
        self.indices = np.arange(n_people)
        self.shuffle = shuffle
        self.counter = 0
        self.num_batches = n_people//batch_size
    def _shuffle(self):
        np.random.shuffle(self.indices)
    def take(self):
        min_index = self.counter*self.batch_size
        max_index = min((self.counter + 1)*self.batch_size, self.n_people)
        self.counter += 1
        indexes =  self.indices[min_index:max_index]
        if self.counter == self.num_batches:
            self.counter = 0
            if self.shuffle:
                self._shuffle()
        return indexes

"""
a = GetIndex(100, 3)
a.take()"
"""


        
        




