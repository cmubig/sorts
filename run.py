# --------------------------------------------------------------------------------------------------
# @file:    run.py
# @brief:   This script is used for running self-play experiments.  
#           Example usage: python run.py --exp path/to/config.json 
# --------------------------------------------------------------------------------------------------
import argparse
import json
import os

from utils.common import Config

def run(exp: str) -> None:
    assert os.path.exists(exp), f"File {exp} does not exist!"

    # load the configuration files
    exp_file = open(exp)
    config = Config(json.load(exp_file))
    
    play = config.SEARCH_POLICY.setting
    if play == "singleagent":
        from game.single_agent import SingleAgent as Game
    elif play == "multiagent":
        from game.multi_agent import MultiAgent as Game
    else:
        raise NotImplementedError(f"Game-type {play} is not Supported!")
    
    game = Game(config)
    game.play()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp', 
        default='./config/multi2_random_mcts.json', 
        type=str, 
        help='path to experiment configuration file')
    args = parser.parse_args()
    
    run(**vars(args))