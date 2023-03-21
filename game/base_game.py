# --------------------------------------------------------------------------------------------------
# @file:    base_game.py
# @brief:   A base class for implementing game modalities for self-play navigation algorithms. 
# --------------------------------------------------------------------------------------------------
import json
import logging
import numpy as np
import os 

from tqdm import tqdm

from gym.gym import Gym
from utils.common import Config, FORMAT
        
class Game:
    """ Runs self-play experiments. """
    def __init__(self, config: Config) -> None:
        """ Intializes game play. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
        """
        self.config = config
        super().__init__()

        self.setup()
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
        
    def play(self) -> None:
        """ Initializes the environment, the policy and run the self-play episodes. """
        self.logger.info(f"Running episodes...")

        # run episodes
        for eps in tqdm(range(self.num_episodes)):
            self.agents = self.gym.spawn_agents()    
            self.gym.reset()
            self.policy.reset()
            self.run_episode(eps)

            metrics = ''
            for m, s in self.metrics.items():
                metrics += f'{m}={s[-1]} '
            self.logger.info(f"Episode [{eps+1}/{self.num_episodes}] - Metrics: {metrics}")
            
        # save metrics
        for m, s in self.metrics.items():
            self.metrics[m] = np.asarray(s).mean()
        metrics_json = json.dumps(self.metrics, indent=4)
        with open(f"{self.out}/metrics.json", 'w') as f:
            f.write(metrics_json)

        self.logger.info(f"Metrics:\n{metrics_json}")
        self.logger.info("Done!")
            
    def run_episode(self, num_episode: int) -> dict:
        """ Runs a single episode. 
        
        Inputs
        ------
        num_episode[int]: id of episode that will run. 
        """
        error_msg = f'Should be implemented by {self.name}'
        raise NotImplementedError(error_msg)
    
    def setup(self) -> None:
        """ Creates the experiment name-tag and all output directories. """
        
        # create the experiment tag name
        exp_name = "exp-{}-{}-{}".format(
            self.config.SEARCH_POLICY.type, 
            self.config.SOCIAL_POLICY.type,
            self.config.GAME.num_agents, 
        )
        
        self.out = os.path.join(self.config.MAIN.out_dir, self.config.DATA.dataset_name, exp_name)
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        assert not self.config.MAIN.sub_dirs == None, f"No sub-dirs were specified!"
        
        self.logger = logging.getLogger(__name__)
  
        output_log = os.path.join(self.out, self.config.MAIN.log_file)
        logging.basicConfig(
            filename=output_log, filemode='a', level=logging.INFO, format=FORMAT, 
            datefmt='%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Initializing Environment...")
        self.gym = Gym(self.config, self.logger, self.out)
                
        if self.config.SEARCH_POLICY.type == "mcts":
            from game.search_policies.mcts import MCTS
            self.policy = MCTS(self.config, self.gym)
        else:
            raise NotImplementedError(f"policy {self.config.SEARCH_POLICY.type} not supported!")
        
        self.num_episodes = self.config.GAME.num_episodes
        self.num_agents = self.config.GAME.num_agents

        self.playing = [0, 1] if self.num_agents > 1 else [0]
        self.agent_rounds = [i for i in range(self.num_agents)]
        self.logger.info(f"{self.name} created output directory: {self.out}")
    