# --------------------------------------------------------------------------------------------------
# @file:    multi_agent.py
# @brief:   Class that inherits from SelfPlay, and implements the procedure for runnning multi-agent
#           episodes for self-play simulation. 
# --------------------------------------------------------------------------------------------------
import numpy as np

from utils.common import Config
from self_play.base_play import SelfPlay
        
class MultiAgent(SelfPlay):
    """ Runs self-play experiments in multi-agent setting. """
    def __init__(self, config: Config) -> None:
        """ Intializes game play. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
        """
        super().__init__(config=config)        
        self.logger.info(f"{self.name} is ready!")

    def run_episode(self, num_episode: int) -> dict:
        """ Runs a single episode. 
        
        Inputs
        ------
        num_episode[int]: id of episode that will run. 

        Outputs
        -------
        metrics[dict]: dictionary containing metric information from the episode.
        """
        step = 0
        current_agent = 0

        while step <= self.config.GAME.max_steps:
            step += 1
            self.policy.reset()

            if self.config.VISUALIZATION.visualize:
                self.gym.show_world(self.agents)
           
            # split playing and non-playing agents
            self.gym.split_agents(self.agents, current_agent)
            
            # compute action probabilities and execute next action
            pi = self.policy.compute_action_probabilities(
                agents=self.agents, current_agent=current_agent)

            if self.config.GAME.deterministic:
                action = np.argmax(pi)
            else: # sample
                action = np.random.choice(self.gym.action_size, size=1, p=pi)[0]

            new_state = self.gym.get_next_state(self.agents[current_agent].trajectory[-1], action)
            self.agents[current_agent].step(new_state)
            
            # check validity of new environment state
            self.agents = self.gym.check_valid(self.agents)
                
            # check if the whole game is done
            if self.gym.active_agents(self.agents) < 1:
                break
            
            # next agent
            current_agent = self.gym.next_agent(self.agents, current_agent)

        for agent in self.agents:
            if not agent.done and step >= self.config.GAME.max_steps:
                agent.update(done=True, timeout=1)
                self.logger.info(f"Agent {agent.id} timed-out!")
                
        # convert visualizations into gif
        if self.config.VISUALIZATION.visualize:
            self.gym.show_world(self.agents)
            self.gym.save(num_episode=num_episode)