# --------------------------------------------------------------------------------------------------
# @file:    agent.py
# @brief:   Implements the Agent class. It contains basic attributes defining the agent's state. 
# --------------------------------------------------------------------------------------------------
import numpy as np 
import torch

from typing import List

class Agent:
    def __init__(
        self, 
        state: np.array, 
        goal_state: np.array, 
        reference_trajectory: np.array, 
        agent_id: int = 0, 
        color: str = 'magenta'
    ) -> None:
        self._state = state
        self._goal_state = goal_state
        self._reference_trajectory = reference_trajectory
        self._color = color
        self._trajectory = [state]
        self._tree = [self._state]
        self._trees = []
        self._id = agent_id
        self._done = False

        self._num_steps = 0
        self._success = 0
        self._collision = 0
        self._offtrack = 0
        self._timeout = 0

    # ----------------------------------------------------------------------------------------------
    # Members
    # ----------------------------------------------------------------------------------------------
    @property
    def num_steps(self) -> int:
        return self._num_steps
    
    @property
    def collision(self) -> int:
        return self._collision
    
    @property
    def offtrack(self) -> int:
        return self._offtrack
    
    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def success(self) -> bool:
        return self._success

    @property
    def done(self) -> bool:
        return self._done    

    @property
    def id(self) -> int:
        return self._id
        
    @property
    def state(self) -> np.array:
        return self._state

    @property
    def goal_state(self) -> np.array:
        return self._goal_state
    
    @property
    def trajectory(self) -> List[np.array]:
        return self._trajectory
    
    @property
    def trees(self) -> List[np.array]:
        return self._trees

    @property
    def tree(self) -> List[np.array]:
        return self._tree

    @property
    def reference_trajectory(self) -> np.array:
        return self._reference_trajectory

    @property
    def color(self) -> str:
        return self._color 
    
    # ----------------------------------------------------------------------------------------------
    # Member setters
    # ----------------------------------------------------------------------------------------------
    def set_collision(self, collision) -> None:
        self._collision = collision

    def set_playing(self, playing) -> None:
        self._playing = playing

    def set_state(self, state) -> None:
        self._state = state

    def set_done(self, done: bool) -> None:
        self._done = done

    def update(
        self, done: bool = False, success: int = 0, collision: int = 0, offtrack: int = 0, 
        timeout: int = 0
    ) -> None:
        self._done = done
        self._success = success
        self._collision = collision
        self._offtrack = offtrack
        self._timeout = timeout

    def add_tree(self) -> None:
        self._trees.append(self._tree)
        self._tree = [self._state]

    def add_tree_state(self, state) -> None:
        self._tree.append(state)
    
    def step(self, state) -> None:
        self._trajectory.append(state)
        self._state = state
        self._tree = []
        self._trees = []
        self._num_steps += 1
        
    def get_location(self, sim: bool = True, scale: int = 1000) -> tuple:
        """ Get agent's current coordinates. """
        state = self.trajectory[-1]
        if sim:
            state = self._state

        x = int(state[-1, 0] * scale)
        y = int(state[-1, 1] * scale)
        z = int(state[-1, 2] * scale)
        return x, y, z
    
    def get_trajectory_np(self) -> np.array:
        trajectory = [s[:-1] for s in self.trajectory[:-1]] + [self.trajectory[-1]]
        return torch.cat(trajectory).numpy()