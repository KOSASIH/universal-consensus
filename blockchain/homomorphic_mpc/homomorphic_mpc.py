import numpy as np
from spdz import SPDZ

class HomomorphicMPC:
    def __init__(self, parties):
        self.parties = parties
        self.spdz = SPDZ()

    def compute(self, inputs):
        # Compute a function on encrypted inputs using SPDZ
        result = self.spdz.compute(self.parties, inputs)
        return result
