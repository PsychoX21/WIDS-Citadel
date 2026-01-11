import numpy as np

class FairValueProcess:
    def __init__(self, initial_value=100.0, sigma=0.5, seed=None):
        self.value = initial_value
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def step(self):
        self.value += self.sigma * self.rng.normal()
        return self.value

    def get(self):
        return self.value
