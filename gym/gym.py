# ------------------------------------------------------------------------------
# @file:    gym.py
# @brief:   Environment implementation. 
# ------------------------------------------------------------------------------
import numpy as np
import torch
import os

import utils.trajair_utils as tu

from logging import Logger
from typing import List, Tuple
from utils.common import Config
from utils.cost_maps import CostMap
from utils.visual import Visual

from utils.trajair_utils import goal_enum
from utils.cost_maps import RUNWAY_ID, X_MIN, X_MAX, Y_MIN, Y_MAX

from gym.agent import Agent

class Gym:
    """ Environment implmenetation. """
    def __init__(self, config: Config, logger: Logger, outdir: str) -> None:
        """ Environment initialization. 
        
        Inputs
        ------
        config[Config]: dictionary containing all configuration parameters. 
        logger[Logger]: logging file. 
        """
        self.config = config
        self.logger = logger
        self.outdir = outdir
        
        self.visual = None
        if self.config.VISUALIZATION.visualize:
            outdir = os.path.join(self.outdir, "vis")
            self.visual = Visual(outdir)
        
        self.dir_start = 3
    
        traj_lib_file = self.config.DATA.trajlib_path
        self.logger.info(f"Setting up action space from: {traj_lib_file}")
        self.traj_lib, self.index_lib = tu.load_action_space(
            traj_lib_file, self.config.DATA.indexlib_path)
        
        reflib_file = self.config.DATA.reflib_path
        self.logger.info(f"Getting reference trajectory from: {reflib_file}")
        self.ref_traj = tu.get_ref_exp_traj(reflib_file)
        
        costmap_path = self.config.DATA.costmap_path
        self.logger.info(f"Creating costmap from: {costmap_path}")
        self.costmap = CostMap(costmap_path)
        self.goal_list = tu.goal_eucledian_list()
        
        self.hh = []

        self.num_init_agents = self.config.GAME.num_agents
        self.playing = []

        super().__init__()

    from gym.start import reset, compute_random_goal, compute_random_start, compute_valid_start_goal, spawn_agents
    from gym.heuristic import get_heuristic_downwind
    from gym.mechanics import get_all_next_states, get_next_state, is_valid, is_done, is_state_space_valid
    from gym.multiagent import split_agents, take_turn, active_agents, next_agent, game_done, is_multi_valid

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def action_size(self) -> int:
        return self.traj_lib.shape[0]

    def get_state_hash(self, agents: List[Agent], sim: bool = True) -> str:
        """ Compute a state key for the current agent. 
        
        Inputs:
        ------
        agents[Agent]: list of agents. 
        sim[bool]: used for querying the agent's location. 

        Outputs
        -------
        state_hash[str]: generated state hash. 
        """
        # there's only one active agent
        if self.active_agents(agents) == 1:
            # for agent in agents:
            #     if not agent.done:
            #         agent_id = agent.id
            #         break
            i = self.playing[0]
            x, y, _ = agents[i].get_location(sim)
            return f'{x}_{y}'
        
        # there's two agents playing
        i, j = self.playing
        xi, yi, _ = agents[i].get_location(sim)
        xj, yj, _ = agents[j].get_location(sim)
        return f'{xi}_{yi}_{xj}_{yj}'
    
    def show_world(self, agents: List[Agent], show: bool = False, show_tree: bool = False) -> None:
        self.visual.reset()
        self.visual.plot(agents, show=show, show_tree=show_tree, pause=1.0)
    
    def save(self, num_episode: int) -> None:
        self.visual.save(num_episode)
        
    def get_cost(self, agents: List[Agent]) -> float:
        """ Get cost at current agent's state. """
        cost = 0.0
        for agent in agents:
            state, goal = agent.state, agent.goal_state
            for i in range(self.dir_start, state.shape[0]):
                x, y, z = state[i, 0].item(), state[i, 1].item(), state[i, 2].item()
                yaw_diff = state[i] - state[i-3]
                slope = np.arctan2(yaw_diff[1], yaw_diff[0])
                wind = -1
                if goal_enum(goal) == RUNWAY_ID:
                    wind = 1
                angle = slope * 180 / np.pi
                if x > X_MIN and x < X_MAX and abs(y) < Y_MAX:
                    cost -= 1

                cost += self.costmap.state_value(x, y, z, angle, wind)
                
        return cost / (state.shape[0]-self.dir_start)