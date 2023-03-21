# --------------------------------------------------------------------------------------------------
# @file:    multi_agent.py
# @brief:   Class that inherits from Game, and implements the procedure for runnning multi-agent
#           episodes for self-play simulation. 
# --------------------------------------------------------------------------------------------------
import numpy as np

from utils.common import Config
from game.base_game import Game
        
class MultiAgent(Game):
    """ Runs self-play experiments in multi-agent setting. """
    def __init__(self, config: Config) -> None:
        """ Intializes game play. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
        """
        super().__init__(config=config)

        # evaluation
        self.metrics = {}
        for metric in ['StepsRate', 'MaxStepsReached', 'SuccessRate', 'CollisionRate']:
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
        current_agent = 0

        while step <= self.config.GAME.max_steps:
            step += 1
            self.policy.reset()

            if len(self.agents) < 1:
                self.logger.info(f"Episode is done!")
                break

            if self.config.VISUALIZATION.visualize:
                self.gym.show_world(self.agents)
           
            # split playing and non-playing agents
            self.gym.split_agents(self.agents)
            
            # compute action probabilities and execute next action
            pi = self.policy.get_action_probabilities(agents=self.agents, current_agent=current_agent)
            
            if self.config.GAME.deterministic:
                action = np.argmax(pi)
            else: # sample
                action = np.random.choice(self.gym.action_size, size=1, p=pi)[0]

            new_state = self.gym.get_next_state(self.agents[current_agent].trajectory[-1], action)
            self.agents[current_agent].update(new_state)
            self.agents[current_agent].add_step()

            # next agent
            current_agent = self.gym.next_agent(self.agents, current_agent)

            # check if the game ended 
            agent_id = self.gym.game_done(self.agents)

            # TODO: handle the case where an agent goes on a completetly wrong direction. 
            if agent_id == -1:
                break
            
            # check if agents are in collision
            multi_valid = self.gym.is_multi_valid(self.agents)
            if multi_valid[0] is not None and multi_valid[1] is not None:
                i, j = multi_valid
                self.logger.info(f"Agents {i} and {j} are in collision; exiting for them!")
                self.agents[i].set_done(True)
                self.agents[i].set_collision(1)

                self.agents[j].set_done(True)
                self.agents[j].set_collision(1)

            # agent with agent_id is done
            if agent_id != None:
                self.agents[agent_id].set_done(True)
                self.agents[agent_id].set_success(1)
                self.logger.info(f"Agent {agent_id} is done!")
                
            # check if the whole game is done
            active_agents = self.gym.active_agents(self.agents)
            if active_agents < 1:
                break

        if step > self.config.GAME.max_steps:
            self.logger.info(f"Episode reached max steps!")
            self.metrics['MaxStepsReached'].append(1)
        else:
            self.metrics['MaxStepsReached'].append(0)
        
        # convert visualizations into gif
        if self.config.VISUALIZATION.visualize:
            self.gym.save(num_episode=num_episode)

        # compute metrics
        for agent in self.agents:
            self.metrics['StepsRate'].append(agent.steps)
            self.metrics['SuccessRate'].append(agent.success)
            self.metrics['CollisionRate'].append(agent.collision)