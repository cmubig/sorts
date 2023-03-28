# --------------------------------------------------------------------------------------------------
# @file:    random_predictor.py
# @brief:   A class for generating a random probability distribution.
# --------------------------------------------------------------------------------------------------
import json 
import numpy as np
import os
import scipy
import torch
from typing import List

import sprnn.utils.common as mutils
from sprnn.trajpred_models.tp_socpatternn import SocialPatteRNN
from utils.common import Config, dotdict
from gym.gym import Agent

class SprnnPolicy:
    """ Implements a socially-aware policy based on the Social-PatteRNN intent prediction algorithm. """
    def __init__(self, config: Config, logger, device: str) -> None:
        self.base_config = config
        self.logger = logger
        self.device = device

        self.setup()
        super().__init__()

    def setup(self) -> None:
        self.num_predictions = self.base_config.SOCIAL_POLICY["num_predictions"]

        model_path = self.base_config.SOCIAL_POLICY["model_config_path"]
        assert os.path.exists(model_path), f"File {model_path} does not exist!"

        # load the configuration files
        self.model_config = dotdict(json.load(open(model_path)))
        self.pred_len = self.model_config.pred_len
        self.pat_len = self.model_config.pat_len
        self.fut_len = self.model_config.fut_len
        self.step = self.model_config.step
        self.k_nearest = self.model_config.model_design["interaction_net"]["k_nearest"]
        self.flatten = True
        self.coord = self.model_config.coord
        
        model_file = self.base_config.SOCIAL_POLICY["model_path"]
        assert os.path.exists(model_file), f"File {model_file} does not exist!"

        model = torch.load(model_file, map_location=torch.device('cpu'))

        # load model 
        self.model = SocialPatteRNN(
            dotdict(self.model_config.model_design), self.logger, self.device).to(self.device)
        self.model.load_state_dict(model['model'])
        self.logger.info(f"Loaded model from: {model_file}.")

    def compute_social_action(self, agents: List[Agent], S: np.array, current_agent: int):
        """ Obtains actions probablities using the Social-PatteRNN prediction algorithm.
        
        Inputs
        ------
        agents[List[Agents]]: list of all playing agents. 
        S[np.array]: set of all possible future states.
        current_agent[int]: ID of agent for which the algorithm computes the social action. 
        """
        # from social-patternn prediction framework:
        states = torch.stack([agent.state for agent in agents]).permute(1, 0, 2)
        H, N, D = states.shape
        patterns = torch.zeros((H, N, self.pat_len-1, D))
        pat = states[:self.pat_len]
        patterns[0] = torch.transpose(pat[1:] - pat[:-1], 0, 1)

        seq_start_end = torch.LongTensor([[0, N]])
  
        soc = mutils.compute_social_influences(
            states[0].unsqueeze(0), pat[-1].unsqueeze(0), seq_start_end, self.k_nearest, self.flatten
        ).to(self.device)
        
        states, patterns = states.to(self.device), patterns.to(self.device)

        _, _, _, h, pat, soc = self.model.evaluate(states, states, patterns, soc, seq_start_end)

        # repeat hidden state and final pattern self.num_samples times to 
        # parallelize inference
        h = h.repeat(1, self.num_predictions, 1)
        pat = pat.repeat(self.num_predictions, 1)
        soc = soc.repeat(self.num_predictions, 1)
        seq_se = torch.LongTensor(
            [[i, i + N] for i in range(0, N * self.num_predictions, N)])
        
        pred = self.model.inference(states[-1], self.fut_len, h, pat, soc, seq_se, self.coord)
        # pred shape is (len, samples, dim)
        pred = pred[:(self.pred_len // self.step), current_agent::N].cpu()

        # convert prediction to action distribution 
        # S shape is (action_size, len, dim)
        S_step = S[:, ::self.step, :]

        # compute the action probability as the error between the set of future states and the 
        # predicted futures:    
        action_probs = np.linalg.norm(pred.unsqueeze(1) - S_step, axis=3)
        action_probs = action_probs.sum(axis=0).sum(axis=1)
        action_probs = action_probs / np.sum(action_probs)
        action_probs = scipy.special.softmax(np.power(action_probs,-1))
        return action_probs
