# --------------------------------------------------------------------------------------------------
# @file:    agent.py
# @brief:   Implements the Agent class. It contains basic attributes defining the agent's state. 
# --------------------------------------------------------------------------------------------------
import numpy as np 

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
        self._tree = []
        self._trees = []
        self._id = agent_id
        self._done = False

        self._steps = 0
        self._success = 0
        self._collision = 0

    # ----------------------------------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------------------------------
    @property
    def steps(self) -> int:
        return self._steps
    
    def add_step(self) -> None:
        self._steps += 1
    
    @property
    def success(self) -> int:
        return self._success
    
    def set_success(self, success) -> None:
        self._success = success
    
    @property
    def collision(self) -> int:
        return self._collision
    
    def set_collision(self, collision) -> None:
        self._collision = collision

    # ----------------------------------------------------------------------------------------------
    # Agent properties
    # ----------------------------------------------------------------------------------------------
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
    
    @property
    def done(self) -> bool:
        return self._done

    def set_playing(self, playing) -> None:
        self._playing = playing

    def set_state(self, state) -> None:
        self._state = state

    def set_done(self, done: bool) -> None:
        self._done = done

    def add_tree(self) -> None:
        self._trees.append(self._tree)
        self._tree = []

    def add_tree_state(self, state) -> None:
        self._tree.append(state)
    
    def update(self, state) -> None:
        self._trajectory.append(state)
        self._state = state
        self._tree = []
        self._trees = []
        
    def get_location(self, sim: bool = True, scale: int = 1000) -> tuple:
        """ Get agent's current coordinates. """
        state = self.trajectory[-1]
        if sim:
            state = self._state

        x = int(state[-1, 0] * scale)
        y = int(state[-1, 1] * scale)
        z = int(state[-1, 2] * scale)
        return x, y, z