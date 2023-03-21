# --------------------------------------------------------------------------------------------------
# @file:    single_agent.py
# @brief:   Class that inherits from Game, and implements the procedure for runnning single-agent
#           episodes for self-play simulation. 
# --------------------------------------------------------------------------------------------------
import numpy as np

from utils.common import Config
from game.base_game import Game
        
class SingleAgent(Game):
    """ Runs self-play experiments. """
    def __init__(self, config: Config) -> None:
        """ Intializes game play. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
        """
        super().__init__(config=config)

        # evaluation
        self.metrics = {}
        for metric in ['Num-Steps', 'MaxStepsReached', 'Success-Rate']:
            self.metrics[metric] = []
        
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
        done = False
        current_agent = 0
        while step < self.config.GAME.max_steps and not done:
            self.policy.reset()
            
            if self.config.VISUALIZATION.visualize:
                self.gym.show_world(self.agents)
            
            pi = self.policy.get_action_probabilities(agents=self.agents, current_agent=current_agent)
            
            if self.config.GAME.deterministic:
                action = np.argmax(pi)
            else: # sample
                action = np.random.choice(self.gym.action_size, size=1, p=pi)[0]

            new_state = self.gym.get_next_state(self.agents[0].trajectory[-1], action)
            self.agents[current_agent].update(new_state)

            # Check if episode is done
            done, _= self.gym.is_done(self.agents, current_agent=current_agent)
            if done:
                self.logger.info(f"Episode {num_episode} is done!")
            
            step += 1
        
        # convert visualizations into gif
        if self.config.VISUALIZATION.visualize:
            self.gym.save(num_episode=num_episode)

        # compute metrics
        self.metrics['Num-Steps'].append(step)
        self.metrics['MaxStepsReached'].append(int(step >= self.config.GAME.max_steps))
        self.metrics['Success-Rate'].append(int(done))