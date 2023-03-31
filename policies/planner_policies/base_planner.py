# --------------------------------------------------------------------------------------------------
# @file:    planner.py
# @brief:   Base class for creating planners. 
# -------------------------------------------------------------------------------------------------- 
from typing import List

from utils.common import Config
from gym.gym import Gym, Agent

class BasePlanner:
    """ Baseline planning algorithm. """
    def __init__(self, config: Config, gym: Gym, logger) -> None:
        """ Intializes the baseline. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
            gym[Gym]: the environment. 
        """
        self.config = config
        self.gym = gym 
        self.logger = logger
        super().__init__()

        self.setup()
        self.logger.info(f"Planner {self.name} is ready!")
        
    # ----------------------------------------------------------------------------------------------
    # Class properties
    # ----------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    # ----------------------------------------------------------------------------------------------
    # Planner methods
    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """ Resets planner parameters. """
        pass

    def setup(self) -> None:
        """ Sets up planner parameters. """
        error_msg = f'Should be implemented by {self.name}'
        raise NotImplementedError(error_msg)
    
    def compute_action_probabilities(self, agents: List[Agent], current_agent: int = 0):
        """ Computes the planner's action probability distribution.
        
        Inputs
        ------
        agents[List[Agent]]: list of playing agents. 
        current_agent[int]: ID of agent for which we'll compute the action probability. 
        """
        error_msg = f'Should be implemented by {self.name}'
        raise NotImplementedError(error_msg)