import numpy as np

class Response:
    def __init__(self, reward, optimal, active_arms=None):
        self.reward = reward
        self.optimal = optimal
        self.active_arms = active_arms