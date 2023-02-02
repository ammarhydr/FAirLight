# FAirLight

We release our constrained reinforcement learning framework for fair and environmentally sustainable traffic signal control. This repository includes the detailed implementation of our FAirLight submitted to KDD 2023

Usage and more information can be found below.

## Installation Guide
### Dependencies
- [PyTorch](https://pytorch.org/get-started/previous-versions/#v120) (The code is tested on PyTorch 1.2.0.) 
- [CityFlowEmission](https://github.com/ammarhydr/CityFlowEmission)
- Tensorflow v1.0+ (for baseline models)

## Example 

Start an example:

``sh run_exp.sh``

Hyperparameters such as learning rate, sample size and the like for the agent can also be assigned in our code and they are easy to tune for better performance.
