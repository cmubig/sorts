# --------------------------------------------------------------------------------------------------
# @file:    multiagent.py
# @brief:   Gym environment multiagent functions.    
# --------------------------------------------------------------------------------------------------
import numpy as np 
from typing import List

from gym.agent import Agent

from utils.common import LOSS_OF_SEPARATION_THRESH
from utils.trajair_utils import direction_goal_detect

def split_agents(self, agents: List[Agent]) -> None:
    """ Splits playing agents """
    active_agents = self.active_agents(agents)
    assert active_agents >= 1, f"No active agents. Can't split them!"
    self.playing = []

    if active_agents == 1:
        for agent in agents:
            if not agent.done:
                self.playing.append(agent.id)
                return
            
    elif active_agents == 2:
        for agent in agents:
            if not agent.done:
                self.playing.append(agent.id)
    # TODO: handle more than 3 agents
    else:
        raise NotImplementedError(f"No suport for num-agents > 2")
    
def take_turn(self, current_agent: int) -> int:
    if len(self.playing) > 1:
        return list(set(self.playing) - set([current_agent]))[0]
    return self.playing[0]

def active_agents(self, agents: List[Agent]) -> int:
    num_agents = 0
    for agent in agents:
        num_agents += int(not agent.done)
    return num_agents

def next_agent(self, agents: List[Agent], current_agent: int) -> int:
    active_agents = self.active_agents(agents)
    assert active_agents >= 1, f"No active agents. Can't roll them!"

    if active_agents == 1:
        for agent in agents:
            if not agent.done:
                return agent.id
    
    # roll next agent
    next_agent = current_agent + 1 if (current_agent + 1) < self.num_init_agents else 0
    while True:
        if not agents[next_agent].done:
            return next_agent
        next_agent += 1

def is_multi_valid(self, agents: List[Agent]):
    """ Checks if the playing agents are in collision. 
    
    Inputs
    ------
    agents[List]: list containing all agents in the scene. 

    Outputs
    -------
    valid[bool]: True if the agent's states are not in collision, otherwise False.
    """
    if self.active_agents(agents) == 1:
        return True, None
    
    for agent in agents:
        if agent.done:
            continue
        agent_start = agent.id
    
    state = agents[agent_start].state 
    for agent in range(agent_start+1, len(agents)):
        difference = state[:, :2] - agents[agent].state[:, :2]
        collision_mask = np.linalg.norm(difference, axis=1) > LOSS_OF_SEPARATION_THRESH
        if not np.all(collision_mask):
            return (agent_start, agent)
    return None, None

def game_done(self, agents: List[Agent], dir_start: int = 3) -> int:
    for agent in agents:
        if agent.done:
            continue
        
        state = agent.state
        goal = agent.goal_state
        for i in range(dir_start, state.shape[0]):
            si, sim1 = state[i], state[i-1]
            dir_s = direction_goal_detect(si, sim1)
            if (dir_s == goal).all():
                return agent.id
            if dir_s.any():
                return -1
    return None
