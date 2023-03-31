# --------------------------------------------------------------------------------------------------
# @file:    multiagent.py
# @brief:   Gym environment multiagent functions.    
# --------------------------------------------------------------------------------------------------
import numpy as np 
from typing import List

from gym.agent import Agent

from utils.common import LOSS_OF_SEPARATION_THRESH, direction_goal_detect, compute_tcpa

def split_agents(self, agents: List[Agent], current_agent: int, time_steps: int = 200) -> None:
    """ Splits playing agents """
    active_agents = self.active_agents(agents)
    assert active_agents >= 1, f"No active agents. Can't split them!"

    if active_agents == 1 or active_agents == 2:
        self.playing = [agent.id for agent in agents if not agent.done]
        
    # TODO: make this efficient
    else:
        state_i = agents[current_agent].trajectory[-1].numpy()

        other_agent = None
        min_tcpa = float('inf')
        for agent in agents:
            if not agent.done and agent.id != current_agent:
                tcpa = compute_tcpa(state_i, agent.trajectory[-1].numpy())
                if tcpa < min_tcpa:
                    min_tcpa = tcpa
                    other_agent = agent.id

        self.playing = [current_agent, other_agent]
    
def take_turn(self, current_agent: int) -> int:
    if len(self.playing) > 1:
        return list(set(self.playing) - set([current_agent]))[0]
    return self.playing[0]

def active_agents(self, agents: List[Agent]) -> int:
    return sum([int(not agent.done) for agent in agents])

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
        next_agent = next_agent + 1 if (next_agent + 1) < self.num_init_agents else 0

def is_multi_valid(self, agents: List[Agent]):
    """ Checks if the playing agents are in collision. 
    
    Inputs
    ------
    agents[List]: list containing all agents in the scene. 

    Outputs
    -------
    valid[bool]: True if the agent's states are not in collision, otherwise False.
    """
    if self.active_agents(agents) > 1:
        for agent in agents:
            if agent.done:
                continue
            agent_start = agent.id

        state = agents[agent_start].state 
        for agent in range(agent_start+1, len(agents)):
            if agents[agent].done:
                continue
            difference = state[:, :2] - agents[agent].state[:, :2]
            collision_mask = np.linalg.norm(difference, axis=1) > LOSS_OF_SEPARATION_THRESH
            if not np.all(collision_mask):
                return False, agent_start, agent
            
    return True, None, None

def check_valid(self, agents: List[Agent], dir_start: int = 3) -> List[Agent]:
    
    # check if two agents are in collision 
    valid, i, j = self.is_multi_valid(agents)
    if not valid:
        agents[i].update(done=True, collision=1)             
        agents[j].update(done=True, collision=1)
        # self.active_agents -= 2
        self.logger.info(f"Agents {i} and {j} are in collision; exiting for them!")

    # check if agents are done
    for agent in agents:
        if agent.done:
            continue
        
        state, goal = agent.trajectory[-1], agent.goal_state
        for i in range(dir_start, state.shape[0]):
            dir_s = direction_goal_detect(state[i], state[i-1])

            # the agent reached its goal
            if (dir_s == goal).all():
                agent.update(done=True, success=1)
                # self.active_agents -= 1
                self.logger.info(f"Agent {agent.id} reached goal!")
                break
            
            # TODO: handle offrack cases
            if dir_s.any():
                agent.update(done=True, offtrack=1)
                # self.active_agents -= 1
                self.logger.info(f"Agent {agent.id} went offtrack; exiting for it!")
                break
                
    return agents