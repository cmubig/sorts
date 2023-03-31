# --------------------------------------------------------------------------------------------------
# @file:    baseline.py
# @brief:   Implements the baseline planning algorithm used in our paper, it simply balances two
#           action distributions: a social action and a reference action using a lambda [0, 1]
#           parameter:
#                       pi = lambda * social + (1 - lambda) * reference
# --------------------------------------------------------------------------------------------------
import torch 
from typing import List

from gym.gym import Gym, Agent
from policies.planner_policies.base_planner import BasePlanner
from utils.common import Config

class Baseline(BasePlanner):
    """ Baseline planning algorithm. """
    def __init__(self, config: Config, gym: Gym, logger) -> None:
        """ Intializes the baseline. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
            gym[Gym]: the environment. 
            logger[logging]: output logger.
        """
        super().__init__(config=config, gym=gym, logger=logger)
    
    # ----------------------------------------------------------------------------------------------
    # Planner-specific methods 
    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """ Resets the planner. """
        pass

    def setup(self) -> None:
        """ Sets up the planner. """
        self.lmbd = self.config.PLANNER_POLICY.lmbd
        assert self.lmbd >= 0.0 and self.lmbd <= 1.0, f"Lambda {self.lmbd} not in range [0, 1]!"
        
        self.device = torch.device('cpu')
        if self.config.MAIN.gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config.MAIN.gpu_id}')

        if self.config.SOCIAL_POLICY.type == "sprnn":
            from policies.social_policies.sprnn import SprnnPolicy
            self.soc_policy = SprnnPolicy(self.config, self.logger, self.device)
        else: 
            raise NotImplementedError(f"Policy {self.config.SOCIAL_POLICY.type} not implemented!")

    def compute_action_probabilities(self, agents: List[Agent], current_agent: int = 0):
        """ Computes the action probabiliy distribution by balancing the social action from the 
            intent prediction model and the reference action from the heuristic function;
                pi = lambda * social_action + (1 - lambda) * reference_action  
        
        Inputs
        ------
            agents[List[Agent]]: dictionary with configuration parameters. 
            current_agent[int]: ID of the agent for which the algorithm is computing the action. 

        Outputs
        -------
            probs[np.array(action_size,)]: numpy array representing the action distribution.
        """
        # compute all next possible states
        S = self.gym.get_all_next_states(agents, current_agent)

        # compute action distribution from social policy
        pi_intent = self.soc_policy.compute_social_action(agents, S, current_agent)

        # compute action distribution from reference heuristic
        pi_reference = self.gym.get_heuristic_downwind(S, agents[current_agent].reference_trajectory)

        # compute final action distribution by balancing the social and reference actions
        return self.lmbd * pi_intent + (1 - self.lmbd) * pi_reference