import numpy as np

class Response:
    def __init__(self, reward, optimal, active=True):
        self.reward = reward
        self.optimal = optimal
        self.active = active