# --------------------------------------------------------------------------------------------------
# @file:    base_play.py
# @brief:   A base class for implementing self-oplay modalities for social navigation algorithms. 
# --------------------------------------------------------------------------------------------------
import json
import logging
import numpy as np
import random
import os 
import time
import torch
from tqdm import tqdm

from gym.gym import Gym
from utils.common import (
    Config, FORMAT, LOSS_OF_SEPARATION_THRESH, compute_reference_error, compute_loss_of_separation)
        
class SelfPlay:
    """ Base class for running self-play experiments. """
    def __init__(self, config: dict) -> None:
        """ Intializes self play. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
        """
        self._config =  Config(config)

        # automated metrics that can be generated after each end of episode
        self.supported_metrics = [
            'Success', 
            'Steps', 
            'Timeout',
            'Offtrack',
            'ReferenceError', 
            'LossOfSeparation_Collision',
            'LossOfSeparation_ClosestDistance',
            'LossOfSeparation_Frames',
        ]
        super().__init__()

        # setup base directory and general parameters
        self.setup()
        self.logger.info(f"Configuration used:\n{json.dumps(config, indent=4)}")
    
    # ----------------------------------------------------------------------------------------------
    # Class properties
    # ----------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def config(self) -> Config:
        return self._config
    
    # ----------------------------------------------------------------------------------------------
    # Base methods, general enough to be used by classes that inherit from this class. 
    # ----------------------------------------------------------------------------------------------
    def setup(self) -> None:
        """ Setup process which includes: 
                * Initialize logger;
                * Create the experiment name-tag;
                * Create output directories, and;
                * Initialize gym, search policy and relevant member parameters. 
        """
        # create the experiment tag name
        exp_name = "policy-{}-{}_num-agents-{}_num-episodes-{}".format(
            self.config.PLANNER_POLICY.type, 
            self.config.SOCIAL_POLICY.type, 
            self.config.GAME.num_agents, 
            self.config.GAME.num_episodes,
        )

        # output directory and logging file
        self.out = os.path.join(self.config.MAIN.out_dir, self.config.DATA.dataset_name, exp_name)
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        assert not self.config.MAIN.sub_dirs == None, f"No sub-dirs were specified!"
        output_log = os.path.join(self.out, self.config.MAIN.log_file)
        logging.basicConfig(
            filename=output_log, filemode='a', level=logging.INFO, format=FORMAT, 
            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"{self.name} created output directory: {self.out}")

        # create environment and planner policy
        self.logger.info(f"Initializing environment and search policy.")
        self.gym = Gym(self.config, self.logger, self.out)
        
        # supported planners
        if self.config.PLANNER_POLICY.type == "mcts":
            from policies.planner_policies.mcts import MCTS as Planner 
        elif self.config.PLANNER_POLICY.type == "baseline":
            from policies.planner_policies.baseline import Baseline as Planner
        else:
            raise NotImplementedError(f"Policy {self.config.PLANNER_POLICY.type} not supported!")
        
        self.policy = Planner(self.config, self.gym, self.logger)
    
        # initialize base parameters
        self.num_episodes = self.config.GAME.num_episodes
        self.num_agents = self.config.GAME.num_agents
        self.playing = [0, 1] if self.num_agents > 1 else [0]
        self.logger.info(f"{self.name} will run {self.num_episodes} {self.num_agents}-agent episodes.")
        
        # metrics cache 
        self.metrics = {}
        for m in self.supported_metrics:
            self.metrics[m] = []
    
        # random seed
        self.seed = self.config.MAIN.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
    
    def play(self) -> None:
        """ Runs self-play episodes, aggregates and saves metrics in the end. """
        self.logger.info(f"Running episodes...")
        start_time = time.time()

        # TODO: parallelize this. 
        # run episodes
        for episode in tqdm(range(self.num_episodes)):
            # reset episode
            self.agents = self.gym.spawn_agents()    
            assert len(self.agents) > 1, f"Invalid number of agents: {len(self.agents)}"
            self.gym.reset()
            self.policy.reset()

            # run episode
            self.run_episode(episode)

            # aggregate episode metrics
            self.aggregate_metrics(episode)
                
        # all episodes are done; save metrics
        for m, s in self.metrics.items():
            self.metrics[m] = float(round(np.asarray(s).mean(), 4))
        metrics_json = json.dumps(self.metrics, indent=4)
        with open(f"{self.out}/metrics.json", 'w') as f:
            f.write(metrics_json)
        self.logger.info(f"Metrics:\n{metrics_json}")

        total_time = time.time() - start_time
        self.logger.info(f"Done! Time: {total_time} (s)")
            
    def aggregate_metrics(self, episode: int) -> None:     
        """ Collect episode metrics. 
         
        Inputs
        ------
            steps[int]: number of steps the episode ran for. 
        """
        # compute metrics
        trajectories = []
        for agent in self.agents:
            self.metrics['Timeout'].append(agent.timeout)
            self.metrics['Steps'].append(agent.num_steps)
            self.metrics['Success'].append(agent.success)   
            self.metrics['Offtrack'].append(agent.offtrack)   
            self.metrics['LossOfSeparation_Collision'].append(agent.collision)
            
            trajectory = agent.get_trajectory_np()
            trajectories.append(trajectory)
            re = compute_reference_error(trajectory, agent.reference_trajectory)
            self.metrics['ReferenceError'].append(re)

        frames, closest_distance = compute_loss_of_separation(
            trajectories, LOSS_OF_SEPARATION_THRESH)
        
        self.metrics['LossOfSeparation_Frames'].append(frames)
        self.metrics['LossOfSeparation_ClosestDistance'].append(closest_distance)

        metrics = ''
        accum_metrics = ''
        for m, s in self.metrics.items():
            metrics += f'{m}={s[-1]} '
            accum_metrics += f'{m}={round(np.asarray(s).mean(), 4)} '
        self.logger.info(f"Episode [{episode+1}/{self.num_episodes}] - Metrics: {metrics}")
        self.logger.info(f"Accumulated Metrics: {accum_metrics}")
    
    # ----------------------------------------------------------------------------------------------
    # Methods below need to be implemented by the child class. 
    # ----------------------------------------------------------------------------------------------
    def run_episode(self, num_episode: int) -> dict:
        """ Runs a single episode. 
        
        Inputs
        ------
        num_episode[int]: id of episode that will run. 
        """
        error_msg = f'Should be implemented by {self.name}'
        raise NotImplementedError(error_msg)