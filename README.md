# rl-course
This is the code repository for programming exercises of the Reinforcement Learning lecture at the University of Stuttgart.
https://ipvs.informatik.uni-stuttgart.de/mlr/reinforcement-learning-ss-20

## Requirements
All exercises will be done with python3.

The first exercise uses numpy and matplotlib:
```bash
python3 -m pip install numpy matplotlib --user
```

Later exercises will use openai gym (https://gym.openai.com/):
```bash
python3 -m pip install gym --user
```

And tensorflow2 for Deep Reinforcement Learning (https://www.tensorflow.org/):
```bash
python3 -m pip install tensorflow --user
```

## Exercises

## Exercise 01 - k-arms Bandit

Epsilon-greedy action selection for a bandit with k-arms.

The Q action-value function is estimated by calculating the expected reward for each action. At each time step, the action that maximizes the Q value function is chosen with probability 1-epsilon (with probability epsilon a random action is chosen).

## Exercise 02 - Brute force value function

In the frozen lake environment, the fixed-point value function is calculated using the Bellman equation for all possible policies. The optimal policy is chosen as the one with maximum value function for all states.

This brute force approach (evaluating all possible policies) is however intractable for large state-action spaces.

## Exercise 03 - Dynamic Programming

Implementation of the Value Iteration algorithm in the Frozen Lake environment

## Exercise 99 - Policy Search

Reinforcement Learning (Policy search), Deep Learning (CNN and Transfer Learning), Image identification (YOLO) and Stochastic Optimization (Genetic algorithm) techniques were used to optimize the policy of the OpenGymAI Lunar-Lander-V2 Environment 

Development made with help of Tensorflow and DEAP packages.