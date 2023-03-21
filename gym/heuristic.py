# --------------------------------------------------------------------------------------------------
# @file:    heuristic.py
# @brief:   Gym heuristic functions used by the planner.    
# --------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp

from utils.common import KM_TO_M, G_ACC

def get_heuristic_downwind(self, states, trajectory):
    """ Computes a heuristic value corresponding to the closest point in the reference trajectory and
    the agent's current state.  
        
    Inputs
    ------
        states[np.array(A, N, L, D)]: the set of all possible future states. 
            A: action space, N: number of agents, L: length of the actions, D: dimensionality
        trajectory[np.array(T, D)]: the reference trajectory for computing the heuristic value. 
            T: length of the trajectory
    
    Output
    ------

    """
    # find the closest point from the reference trajectory to the agent's first step.
    s_0 = states[0, 0, :]
    idx_closest = np.argmin(
        np.linalg.norm(trajectory[:-2] - np.tile(s_0, (trajectory.shape[0]-2, 1)), axis=1))
    
    # NOTE: what on earth is going on here? what are these magic numbers for? 30, 19, 5333, 
    idx = min(idx_closest + 30, trajectory.shape[0] - 1)
    compare_point = min(idx - idx_closest, 19)
    idx_min = idx - compare_point

    # NOTE: does this mean downwind velocity? 
    vel_d = np.mean(
        np.linalg.norm(trajectory[1+idx_min:idx, :2]-trajectory[idx_min:idx-1, :2],axis=1)) * KM_TO_M
    vert_d = (np.mean(trajectory[1+idx-compare_point:idx, 2:]) - s_0[2].numpy()) * 5333 #-traj[idx-compare_point:idx-1, 2:])*196850

    # # ang1 = np.arctan2(curr_position[:, 0, compare_point, 1]-curr_position[:, 0, compare_point-1, 1],curr_position[:, 0, compare_point, 0]-curr_position[:, 0, compare_point-1, 0])
    # # ang2 = np.arctan2(traj[idx, 1]-curr_position[0,0,0,1],traj[idx, 0]-curr_position[0,0,0,0])
    # # print(curr_position.shape)
    # # ang_diff = ang1-ang2
    # # ang_diff[ang_diff<0] += 2*np.pi
    # # dir_h = np.argmin(ang_diff)
    # ang_d = self.index_lib[dir_h][2] + 1e-6

    ang1 = np.arctan2(
        states[0, compare_point, 1]-states[0, compare_point-1, 1],
        states[0, compare_point, 0]-states[0, compare_point-1, 0])
    ang2 = np.arctan2(trajectory[idx, 1] - s_0[1], trajectory[idx, 0] - s_0[0])
    ang_diff = ang1-ang2
    ang_diff = ang_diff if ang_diff>0 else ang_diff + 2*np.pi
    L1 = np.linalg.norm((trajectory[idx,:2]) - states[0,0,:2].numpy()) * KM_TO_M
    ang_d = -np.arctan2(2*np.sin(ang_diff)*vel_d**2,(L1*G_ACC)).numpy()
    # angles = np.deg2rad(np.array([2,7,15,30]))
    # ang_d = np.sign(ang_d)*angles[np.argmin(abs(angles-abs(ang_d)))]

    # print(ang_d*180/np.pi)
    vel_w = 100
    ang_w = 1000
    vert_w = 20
    # ang_w = 200
    # vert_w = 10

    vel_h = vel_w*(self.index_lib[:,1] - vel_d)
    ang_h = ang_w*(self.index_lib[:,2] - ang_d)
    vert_h = vert_w*(self.index_lib[:,3] - vert_d)

    heuristic = np.linalg.norm(np.stack((vel_h,ang_h,vert_h)),axis=0)
    heuristic = heuristic / np.sum(heuristic)   
    return sp.special.softmax(np.power(heuristic, -1))