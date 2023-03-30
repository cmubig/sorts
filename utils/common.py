# ------------------------------------------------------------------------------
# @file:    common.py
# @brief:   This file contains the implementation common utility classes and 
#           function needed by the modules in sorts.
# ------------------------------------------------------------------------------
import numpy as np 
import logging
import numpy as np
import os
import pickle
import torch

from typing import Tuple, Any
from itertools import combinations

EPS = 1e-8
FORMAT = '[%(asctime)s: %(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
KM_TO_M = 1000
G_ACC = 9.81
LOSS_OF_SEPARATION_THRESH = 0.3
THRESH = 12 #KM
RADIUS = 10 # Km
DIR_LOOKUP = {
    0: 'W', 45: 'SW', 90: 'S', 135: 'SE', 180: 'E', 225: 'NE', 270: 'N', 315: 'NW'
}

COLORS = ['dodgerblue', 'darkorange', 'mediumseagreen', 'mediumblue']


class Config:
    """ A class for holding configuration parameters. """
    def __init__(self, config):
        self.MAIN = dotdict(config)

        self.DATA = None
        if self.MAIN.data:
            self.DATA = dotdict(self.MAIN.data)
        
        self.GAME = None 
        if self.MAIN.game:
            self.GAME = dotdict(self.MAIN.game)
        
        self.PLANNER_POLICY = None
        if self.MAIN.planner_policy:
            self.PLANNER_POLICY = dotdict(self.MAIN.planner_policy)
        
        self.SOCIAL_POLICY = None
        if self.MAIN.social_policy:
            self.SOCIAL_POLICY = dotdict(self.MAIN.social_policy)

        self.VISUALIZATION = None
        if self.MAIN.visualization:
            self.VISUALIZATION = dotdict(self.MAIN.visualization)

        self.METRICS = None
        if self.MAIN.metrics:
            self.METRICS = dotdict(self.MAIN.metrics)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def compute_reference_error(trajectory: np.array, reference: np.array) -> float:
    # TODO: use dynamic time warping instead?
    re_0 = reference[0]
    tr_0 = np.argmin(np.linalg.norm(trajectory - re_0, axis=1))
    trajectory = trajectory[tr_0:]
    if len(trajectory) > len(reference):
        trajectory = trajectory[:len(reference)]
    else:
        reference = reference[:len(trajectory)]
    return np.linalg.norm(trajectory - reference)

def compute_loss_of_separation(trajectories: list, threshold: float, mean: bool = True) -> float:
    list_agents = list(range(len(trajectories)))
    ls = []
    cd = []
    for i, j in combinations(list_agents, 2):
        traj_i, traj_j = trajectories[i], trajectories[j]
        if len(traj_i) > len(traj_j):
            traj_i = traj_i[:len(traj_j)]
        else:
            traj_j = traj_j[:len(traj_i)]

        # compute distance
        dist = np.linalg.norm(traj_i - traj_j, axis=1)
        cd.append(dist.min())
        ls.append((dist <= threshold).sum())

    if mean:
        return np.asarray(ls).mean(), np.array(cd).mean()
    return ls, cd

def compute_tcpa(state_i: np.array, state_j: np.array, time_steps: int = 200):

    steps = np.arange(1, time_steps)

    si_diff = state_i[-1, :2] - state_i[-2, :2]
    si_fut = state_i[-1, :2][None, :] + np.multiply(steps[:, None], si_diff[None, :])
    
    sj_diff = state_j[-1, :2] - state_j[-2, :2]
    sj_fut = state_j[-1, :2][None, :] + np.multiply(steps[:, None], sj_diff[None, :])
    
    tcpa = np.argmin(np.linalg.norm(si_fut - sj_fut, axis=1))
    return tcpa

def load_action_space(
    lib_path: str, index_path: str, scale: float = 1000, 
    action_space: list = [252, 3, 20], index_space: list = [252, 6]
) -> Tuple[np.array, np.array]:
    """ Loads the action space from a trajectory lib. 
    
    Inputs
    ------ 
    lib_path[str]: path to the trajectory library containing the actions.
    index_path[str]: path to the action index file
    scale[float]: scaling factor for the action. 
    action_space[list]: action space size. 
    
    Outputs
    -------
    traj_lib[np.array]: actions
    index_lib[np.array]: actions indeces
    """
    traj_lib = np.zeros(action_space)
    index_lib = np.zeros(index_space)
    j = 0
    for count, line in enumerate(open(lib_path, 'r',newline='\n')):
        if count%4 == 1:
            traj_lib[j,0] = np.fromstring(line.strip(), dtype=float, sep=' ')
        if count%4 == 2:
            traj_lib[j,1] = np.fromstring(line.strip(), dtype=float, sep=' ')
        if count%4 == 3:
            traj_lib[j,2] = np.fromstring(line.strip(), dtype=float, sep=' ')
            j+=1
    traj_lib = np.stack(traj_lib, axis=0) / scale
    
    for i, line in enumerate(open(index_path, 'r',newline='\n')):
        index_lib[i, :] = np.fromstring(line.strip(), dtype=float, sep=' ' )
    index_lib = np.stack(index_lib, axis=0)
    
    return traj_lib, index_lib

def get_ref_exp_traj(reflib_path: str) -> Any:
    """ Loads the reference trajectories. 
    
    Inputs
    ------
    reflib_path[str]: path to the reference trajectory library. 
    """
    if not os.path.exists(reflib_path):
        logging.info("No ref libs found!")
        return None
    
    with open(reflib_path,'rb') as f:
        traj = pickle.load(f)
        return traj

def goal_eucledian_list(num_goals: int = 10) -> list:
    """ Gets a list of goals.  
    
    Inputs
    ------
    num_goals[int]: number of goals to create.
    """
    ang = np.array([90, 45, 0, -45, -90, -135, 180, 135])
    pos = []
    for goal_idx in range(num_goals): 
        
        if goal_idx < 8:
            pos.append(
                np.array([(THRESH+1)*np.cos(np.deg2rad(ang[goal_idx])),
                          (THRESH+1)*np.sin(np.deg2rad(ang[goal_idx])), 1.0]))
        elif goal_idx == 8:
            pos.append(np.array([0.0, 0.0, 0.2]))
        elif goal_idx == 9:
           pos.append(np.array([-3.0, -3.5, 0.8]))

    return pos


def direction_goal_detect(pos,second_pos):
    
    dir_array = torch.zeros([10]) ## [N, NE, E, SE, S, SW, W, NW, R1, R2]
    yaw_diff = pos-second_pos

    if np.linalg.norm(pos) > THRESH  :
        # print("diff",difference,'pos',pos, "input_pos",input_pos)
            planar_slope = torch.atan2(pos[1],pos[0])
            degrees_slope = planar_slope*180.0/np.pi
          

            if degrees_slope <22.5 and degrees_slope >-22.5: #east
                dir_array[2] = 1.0
            elif degrees_slope <67.5 and degrees_slope >22.5: #NE
                dir_array[1] = 1.0
            elif degrees_slope <112.5 and degrees_slope >67.5: #N
                dir_array[0] = 1.0
            elif degrees_slope <157.5 and degrees_slope >112.5: # NW
                dir_array[7] = 1.0
            elif degrees_slope <-157.5 or degrees_slope >157.5: # W
                dir_array[6] = 1.0
            elif degrees_slope <-22.5 and degrees_slope >-67.5: #SE
                dir_array[3] = 1.0
            elif degrees_slope <-67.5 and degrees_slope >-112.5: #S
                dir_array[4] = 1.0
            elif degrees_slope <-112.5 and degrees_slope >-157.5: #SW:
                dir_array[5] = 1.0
        # print("Outer pos reached",goal_enum(dir_array))
    else:
        
            yaw_diff_slope = torch.atan2(yaw_diff[1],yaw_diff[0])
            yaw_diff_slope_degrees = yaw_diff_slope*180.0/np.pi
            # print(yaw_diff_slope_degrees)
            if pos[0]<0.2 and pos[0]> -0.2 and abs(pos[1])<0.20 and pos[2] <0.3: #1
                if abs(yaw_diff_slope_degrees) <20.0:
                    dir_array[8] = 1.0
                    return dir_array
                    # print("Runway reached",goal_enum(dir_array))


            elif pos[0]<1.7 and pos[0]> 1.5 and abs(pos[1])<0.2:
                # print("no alt")
                if pos[2] <0.5:  #2,
                # print("bad head",abs(yaw_diff_slope_degrees))

                    if 180-abs(yaw_diff_slope_degrees) <20.0:
                        dir_array[9] = 1.0
                        # print("good head")

                        return dir_array
                        # print("Runway reached",goal_enum(dir_array))

    return dir_array

def goal_enum(goal):
    msk = goal.squeeze().numpy().astype(bool)
    g = ["N","NE","E","SE","S","SW","W","NW","R1","R2"]
    return [g[i] for i in range(len(g)) if msk[i]]