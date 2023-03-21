# --------------------------------------------------------------------------------------------------
# @file:    start.py
# @brief:   Gym utilities for generating the agents' inital conditions. 
# --------------------------------------------------------------------------------------------------
import numpy as np 
import torch

from scipy.spatial.transform import Rotation as R

from gym.agent import Agent
from utils.trajair_utils import DIR_LOOKUP, RADIUS, COLORS

def reset(self) -> None:
    """ Reset environment and visuals. """
    self.visual.reset()

def spawn_agents(self, angle_step: int = 45) -> None:
    """ Spawns agents in the environment. 
    
    Inputs
    ------
    angle_step[int]: angle step for getting the possible orientations.

    Outputs
    -------
    agents[List[Agent]]: list of agents spawend in the environment.
    """
    angle_deg = np.random.choice(
        np.arange(0, 360, angle_step), self.num_init_agents, replace=False)
    
    agents = []
    for i in range(self.num_init_agents):
        state, goal, ref_trajectory = self.compute_valid_start_goal(angle_deg[i])
        # self.logger.info(f"Spawned agent {i} at: {state}: {state.shape}")
        # self.logger.info(f"Goal for agent {i} is at {goal}")
        agents.append(Agent(state, goal, ref_trajectory, i, COLORS[i]))
        
    return agents

def compute_random_start(self, angle_deg: float, z: float = 0.8):
    """ Randomly generates a random start position from a given angle and height and the trajectory
    library. 
    
    Inputs
    ------
    angle_deg[float]: angle from which to sample the position. 
    z[float]: height from which to sample the start position. 

    Outputs
    -------
    start[torch.tensor]: starting trajectory.
    """
    angle = np.deg2rad(angle_deg)
    x = RADIUS * np.cos(angle + np.pi)
    y = RADIUS * np.sin(angle + np.pi)
    dir_matrix = R.from_euler('z', angle).as_matrix()

    trajs = (np.dot(dir_matrix, self.traj_lib[2]) + (np.array([x,y,z])[:,None])).T
    
    return torch.from_numpy(trajs).float()

def compute_random_goal(self, num_goals: int = 10, p = None):
    """ Randomly generates a random goal location from a distributions of goals.
    
    Inputs
    ------
    num_goals[int]: number of goals to sample from.
    
    Outputs
    -------
    goal[torch.tensor]: one-hot tensor representing the goal. 
    """
    return torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1, p = p)]).float()

def compute_valid_start_goal(self, angle_deg: float, runway: str = 'R2'):
    """ Generates a valid start, goal and refrence trajectory. 
    
    Inputs
    ------
    angle_deg[float]: angle from which to sample the position. 
    
    Outputs
    -------
    start[torch.tensor]: starting trajectory. 
    goal[torch.tensor]: one-hot tensor. 
    ref_traj[torch.tensor]: reference trajectory given the start and goal.
    """
    start_position = self.compute_random_start(angle_deg)
    goal = self.compute_random_goal(p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        
    if self.ref_traj is not None:
        ref_traj = self.ref_traj[(DIR_LOOKUP[angle_deg], runway)].to_numpy()
    
    return start_position, goal, ref_traj