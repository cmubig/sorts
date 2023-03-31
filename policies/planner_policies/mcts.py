# --------------------------------------------------------------------------------------------------
# @file:    mcts.py
# @brief:   Implements the Monte Carlo Tree Search (MCTS) algorithm. 
# --------------------------------------------------------------------------------------------------
import numpy as np
import time
import torch 
from math import sqrt
from typing import List

from gym.gym import Gym, Agent
from policies.planner_policies.base_planner import BasePlanner
from utils.common import Config, EPS

class MCTS(BasePlanner):
    """ Monte Carlo Tree Search algorithm. """
    def __init__(self, config: Config, gym: Gym, logger) -> None:
        """ Intializes MCTS. 
        
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
        self.Q_sa = {}
        self.N_sa = {}
        self.N_s = {}
        self.P_s = {}
        self.E_s = {}
        self.V_s = {}
        self.H_s = {}
        self.H_sa = {}

    def setup(self) -> None:
        """ Sets up the planner. """
        self.c_uct = self.config.PLANNER_POLICY.c_uct
        self.h_uct = self.config.PLANNER_POLICY.h_uct
        
        self.device = torch.device('cpu')
        if self.config.MAIN.gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config.MAIN.gpu_id}')

        if self.config.SOCIAL_POLICY.type == "sprnn":
            from policies.social_policies.sprnn import SprnnPolicy
            self.soc_policy = SprnnPolicy(self.config, self.logger, self.device)
        else: 
            raise NotImplementedError(f"Policy {self.config.SOCIAL_POLICY.type} not implemented!")
        
        # stores Q values for state, action (s, a), as defined in the paper
        self.Q_sa = {}
        
        # stores number of times edge (s, a) was visited
        self.N_sa = {}
        
        # stores number of times a board s was visited 
        self.N_s = {}
        
        # stores initial policy (returned by the neural net)
        self.P_s = {}
        
        # stores if game ended for board s
        self.E_s = {}
        
        # stores valid moves for board s
        self.V_s = {}
        
        # stores heuristic value for s
        self.H_s = {}
        
        # stores heuristic value for (s, a)
        self.H_sa = {}

    def compute_action_probabilities(self, agents: List[Agent], current_agent: int = 0):
        """ Runs tree search simulations for a given time or number of simulations, and computes the 
        action probabilities based on the state visitation count computed via the simulations. 
        
        Inputs
        ------
            agents[List[Agent]]: dictionary with configuration parameters. 
            gym[Gym]: the environment. 

        Outputs
        -------
            probs[np.array(action_size,)]: numpy array with the action probability distribution.
        """
        start_time = time.time()

        # time-based search
        if self.config.PLANNER_POLICY.search == "time":
            while (time.time() - start_time) < self.config.PLANNER_POLICY.max_time:
                self.search(agents, current_agent=current_agent)
        # tree expansions-based search
        else:
            for t in range(self.config.PLANNER_POLICY.num_ts):
                self.search(agents, current_agent=current_agent)
                # TODO: reset state and tree for next expansion ?
                for agent in agents:
                    agent.set_state(agent.trajectory[-1])

                # agents[current_agent].set_state(agents[current_agent].trajectory[-1])
                # agents[current_agent].add_tree()
                
        state = self.gym.get_state_hash(agents=agents, sim=False)
        
        counts = [
            self.N_sa[(state, action)] if (state, action) in self.N_sa else 0 for action in range(
                self.gym.action_size)]
        
        counts_sum = float(sum(counts))
        if int(counts_sum) != 0:
            # return [x / counts_sum for x in counts]
            return np.asarray(counts) / counts_sum
        
        if self.config.VISUALIZATION.visualize:
            self.gym.show_world(agents, show_tree=True)
        
        return np.ones_like(counts) / self.gym.action_size
        
    def search(self, agents, heuristic: float = 0, current_agent: tuple = (0, 0)) -> None:
        """ Runs tree search for the playing agents. 
        
        Inputs
        ------
            agents[List[Agent]]: dictionary with configuration parameters. 
            heurisic[np.array(action_size,)]: probability distribution for the reference heuristic. 
            current_agent[int]: index of the current playing agent.

        Outputs
        -------
            v[float]: expected value for the tree search.
            h[np.array(action_size,)]: reference heuristic value. 
        """
        state = self.gym.get_state_hash(agents=agents)

        if state not in self.E_s:
            self.E_s[state], _ = self.gym.is_done(agents, current_agent=current_agent)
        
        # terminal node
        if self.E_s[state] != 0:
            return self.E_s[state], heuristic
        
        # get the set of all possible next states
        S = self.gym.get_all_next_states(agents, current_agent)
        
        # check if state in prediction policy 
        if state not in self.P_s:
            v_s = self.gym.get_cost(agents)

            self.N_s[state] = 0
            self.P_s[state] = self.soc_policy.compute_social_action(agents, S, current_agent) 

            # if self.config.VISUALIZATION.visualize:
            #     self.gym.show_world(agents, show_tree=True, agent_id=current_agent)
            
            return v_s, heuristic
        
        current_best = -float('inf')
        best_action = -1
        heuristic = self.gym.get_heuristic_downwind(S, agents[current_agent].reference_trajectory)
        
        S_valid = self.gym.is_state_space_valid(S, agents, current_agent)

        for action in range(self.gym.action_size):
            if S_valid[action]:
                if (state, action) in self.Q_sa:
                    n = sqrt(self.N_s[state]) / (1 + self.N_sa[(state, action)])
                    uct = (
                        self.Q_sa[(state, action)] 
                        + self.c_uct * self.P_s[state][action] * n 
                        + self.h_uct * self.H_sa[(state, action)])
                else:
                    uct = (
                        self.c_uct * self.P_s[state][action] * sqrt(self.N_s[state] + EPS) 
                        + self.h_uct * heuristic[action])
                
                if uct > current_best:
                    current_best = uct 
                    best_action = action
        
        action = best_action
       
        new_state = self.gym.get_next_state(agents[current_agent].state, action)
        agents[current_agent].set_state(new_state)
        agents[current_agent].add_tree_state(new_state)

        current_agent = 0 if len(agents) == 1 else self.gym.take_turn(current_agent)
        v, h = self.search(agents, heuristic[action], current_agent)

        if (state, action) in self.Q_sa:
            self.Q_sa[(state, action)] = (
                (self.N_sa[(state, action)] * self.Q_sa[(state, action)] 
                 + v) / (self.N_sa[(state, action)] + 1))
            
            self.N_sa[(state, action)] += 1
            self.H_sa[(state, action)] = (
                (self.N_sa[(state, action)] * self.H_sa[(state, action)] 
                 + h) / (self.N_sa[(state, action)] + 1))
        else:
            self.Q_sa[(state, action)] = v
            self.N_sa[(state, action)] = 1
            self.H_sa[(state, action)] = h

        self.N_s[state] += 1
        return v, h