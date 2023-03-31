# --------------------------------------------------------------------------------------------------
# @file:    random_predictor.py
# @brief:   A class for generating a random probability distribution.
# --------------------------------------------------------------------------------------------------
import numpy as np

from scipy.stats import norm
from typing import List

from utils.common import Config
from gym.gym import Agent

class BaseSocialPolicy:
    """ Base class for implementing social policies. """
    def __init__(self, config: Config, logger, device: str) -> None:
        self.config = config
        self.logger = logger
        self.device = device

        self.num_predictions = self.config.SOCIAL_POLICY["num_predictions"]
        super().__init__()

    # ----------------------------------------------------------------------------------------------
    # Class properties
    # ----------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    # ----------------------------------------------------------------------------------------------
    # Social policy methods
    # ----------------------------------------------------------------------------------------------
    def setup(self) -> None:
        """ Sets up planner parameters. """
        error_msg = f'Should be implemented by {self.name}'
        raise NotImplementedError(error_msg)
    
    def compute_social_action(self, agents: List[Agent], current_agent: int = 0):
        """ Computes the planner's action probability distribution.
        
        Inputs
        ------
        agents[List[Agent]]: list of playing agents. 
        current_agent[int]: ID of agent for which we'll compute the action probability. 
        """
        error_msg = f'Should be implemented by {self.name}'
        raise NotImplementedError(error_msg)