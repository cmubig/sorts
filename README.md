# [Refactoring in progress] Social Robot Tree Search (SoRTS)

This repository contains the code for the paper:

<h3> 
Learned Tree Search for Long-Horizon Social Robot Navigation in
Shared Airspace
</h3>

[Ingrid Navarro](https://navars.xyz) *, [Jay Patrikar](https://www.jaypatrikar.me) *, Joao P. A. Dantas, 
Rohan Baijal, Ian Higgins, Sebastian Scherer and [Jean Oh](https://www.cs.cmu.edu/~./jeanoh/) 

<p align="center">
  <img width="600" src="./readme/abstract.png" alt="SoRTS">
</p>

## Model Overview

Social Robot Tree Search (SoRTS) is an algorithm for the safe navigation of mobile robots in social 
domains. SoRTS aims to augment existing socially-aware trajectory prediction policies with a Monte 
Carlo Tree Search (MCTS) planner for improved downstream navigation of mobile robots. 
<p align="center">
  <img width="1000" src="./readme/model.png" alt="SoRTS">
</p>

## Installation

Setup a conda environment:
```
conda create --name sorts python=3.9
conda activate sorts
```

Download the repository and install requirements:
```
git clone --branch sprnn git@github.com:cmubig/sorts.git
cd sorts
pip install -e . 
```

## Dataset

TODO

## Running the code

TODO

#### Running the SocialPatteRNN model

TODO

## Results

TODO

## Citing

#### TODO: update this
```tex
@inproceedings{name,
  title={Paper},
  author={Author1 and Author2},
  booktitle={Conference},
  year={2022}
 }
```
