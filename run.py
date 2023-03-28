# --------------------------------------------------------------------------------------------------
# @file:    run.py
# @brief:   This script is used for running self-play experiments by providing a configuration file.  
#           Example usage: python run.py --exp path/to/config.json 
# --------------------------------------------------------------------------------------------------
import argparse
import json
import os

def run(exp: str) -> None:
    assert os.path.exists(exp), f"File {exp} does not exist!"

    # load the configuration files
    config = json.load(open(exp))

    play = config['planner_policy']['setting']
    if play == "multiagent":
        from self_play.multi_agent import MultiAgent as SelfPlay
    else:
        raise NotImplementedError(f"Self play of type {play} is not supported!")
    
    game = SelfPlay(config)
    game.play()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp', 
        default='./config/sprnn_mcts/2agents.json', 
        type=str, 
        help='path to experiment configuration file')
    args = parser.parse_args()
    
    run(**vars(args))