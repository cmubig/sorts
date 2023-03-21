# --------------------------------------------------------------------------------------------------
# @file:    random_predictor.py
# @brief:   A class for generating a random probability distribution.
# --------------------------------------------------------------------------------------------------
import numpy as np

from scipy.stats import norm
from typing import List

from utils.common import Config
from gym.gym import Agent

class RandomPredictor:
    """ Implements a random probability distribution predictor. """
    def __init__(self, config: Config) -> None:
        self.config = config

        self.num_predictions = self.config.SOCIAL_POLICY["num_predictions"]
        super().__init__()
        
    def get_social_action(self, agents: List[Agent], S: np.array):
        """ Obtains actions probablities using random sampling.
        
        Inputs
        ------
        num_episode[int]: id of episode that will run. 
        """
        action_size = S.shape[0]

        action_probs = np.zeros(shape=(action_size, self.num_predictions))
        for i in range(self.num_predictions):
            rvs = norm.rvs(size=action_size)
            ap = rvs - np.min(rvs)
            ap /= np.sum(ap)
            action_probs[:, i] = ap

        return action_probs.mean(axis=1)