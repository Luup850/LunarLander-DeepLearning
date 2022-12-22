# LunarLander-DeepLearning

This repository contains the code to train three different models on the [lunarLander-v2 environment](https://www.gymlibrary.dev/environments/box2d/lunar_lander/). The three models are:
- Policy Gradient
- Deep Q Learning
- Double Deep Q



## To run the code
> Note: the code has been tested using Python3.9.11

First, create a virtual environment on python and install the dependencies from the [requirements file](requirements.txt). This can be done with the following command:

`python3 -m pip install -r requirements.txt`

Then to train the models run the notebook [LunarLander_Final.ipynb](/Project/LunarLander_Final.ipynb).

## About the [LunarLander_Final.ipynb](/Project/LunarLander_Final.py)

The notebook is designed to train the three different models, save them and generate a plot for each with the average and the per step score.

Additionally, [LunarLander.py](/Project/LunarLander_Final.py) contains the same code in the form of a `.py`. This is better to run it on a system without graphical interface like an hpc.

A trained model for each can be found in [models](/models/), and the corresponding plots in [plots](/plots/). Next, the three results are plotted.

### Gradient descent
![Gradient Descent](/plots/Policy_Gradient_training_rewards.png)

### Deep Q
![Deep Q](/plots/DQN_training_rewards.png)

### Double Deep Q
![Double Deep Q](/plots/DDQN_training_rewards.png)

