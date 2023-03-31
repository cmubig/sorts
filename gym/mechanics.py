# --------------------------------------------------------------------------------------------------
# @file:    mechanics.py
# @brief:   Gym environment mechanics functions.    
# --------------------------------------------------------------------------------------------------
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import List

from gym.agent import Agent
from utils.common import direction_goal_detect, LOSS_OF_SEPARATION_THRESH

def is_done(self, agents: Agent, current_agent: int) -> bool:
    """ Checks if agent is done. 
    
    Inputs
    ------
    agents[Agent]: list of agents in the episode.
    current_agent[int]: ID of the current playing agent. 


    Outputs
    -------
    valid[int]: the validity of the state: -1 collision, 1: done, 0: not done.
    dir[float]: the direction of the state.
    """
    # Check if agents are colliding
    if self.active_agents(agents) > 1 and not self.is_valid(agents):
        return -1, None

    state, goal = agents[current_agent].state, agents[current_agent].goal_state
    for t in range(self.dir_start, state.shape[0]):
        s_dir = direction_goal_detect(state[t], state[t-1])
        if (s_dir == goal).all():
            return 1, s_dir
        if s_dir.any():
            return 0, s_dir
    return 0, s_dir

def is_valid(self, agents: List[Agent]) -> bool:
    """ Checks if the playing agents are in collision. 
    
    Inputs
    ------
    agents[List]: list containing all agents in the scene. 

    Outputs
    -------
    valid[bool]: True if the agent's states are not in collision, otherwise False.
    """
    if self.active_agents(agents) == 1:
        return True

    i, j = self.playing
    state_i, state_j = agents[i].state, agents[j].state

    difference = state_i[:, :2] - state_j[:, :2]
    collision_mask = np.linalg.norm(difference, axis=1) > LOSS_OF_SEPARATION_THRESH
    return np.all(collision_mask)

def is_state_space_valid(self, states: np.array, agents: List[Agent], current_agent: int):
    """ Compute all next states for the agent with agent_id
    
    Inputs
    ------
    states[np.array]: the set of all possible next states for the current agent.
    agents[List]: list containing all agents in the scene. 
    current_agent[int]: ID of the agent for which to compute the next possible states. 

    Outputs
    -------
    valid[np.array(bool)]: boolean array where True corresponds to a valid state and False, invalid.
    """
    action_space, traj_len, dim = states.shape
    if self.active_agents(agents) == 1:
        # If there's one agent, let's assume all states are valid
        return np.ones((action_space, ), dtype=bool)
    else:
        # get id and state of non-playing agent
        other_agent = list(set(self.playing) - set([current_agent]))[0]
        states_other = agents[other_agent].state.repeat(action_space, 1, 1)

        # check if any of the playing agent's possible state are in collision with the other agent's
        # current state
        difference = states[:, :, :2] - states_other[:, :, :2]
        collision_mask = np.linalg.norm(difference, axis=1) > LOSS_OF_SEPARATION_THRESH
        return np.all(collision_mask, axis=1)

def get_all_next_states(self, agents: List[Agent], agent_id: int):
    """ Compute all next states for the agent with agent_id
    
    Inputs
    ------
    agents[List]: list containing all agents in the scene. 
    agent_id[int]: ID of the agent for which to compute the next possible states. 

    Outputs
    -------
    trajs[List[Agent]]: list of agents spawend in the environment.
    """
    state = agents[agent_id].state
    difference = state[-1] - state[-3]
    angle = np.arctan2(difference[1], difference[0])
    
    # direction matrix
    rot = np.squeeze(R.from_euler('z', angle).as_matrix())
    
    next_states = np.matmul(rot[None, :], self.traj_lib) + np.array(state[-1, :])[None, :, None]
    return torch.transpose(torch.from_numpy(next_states), 2, 1)

def get_next_state(self, state, action):
    """ Computes the next state given an action choice by rotating and translating the corresponding 
    motion primitive to the end of previous executed trajectory. 
    
    Inputs
    ------
    state[torch.tensor]: agent's current state. 
    action[int]: action to execute. 

    Outputs
    -------
    next_state[torch.tensor]: 
    """
    difference = (state[-1, :] - state[-3, :]).cpu().numpy()
    angle = np.arctan2(difference[1], difference[0])

    # direction matrix
    rot = np.squeeze(R.from_euler('z', angle).as_matrix())

    next_state = (rot @ self.traj_lib[action] + np.array(state[-1][:, None])).T
    return torch.from_numpy(next_state).float()