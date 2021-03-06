import sys
import os

import envs
import numpy as np
import time 
import imageio
import copy
from tqdm import tqdm

def main(run_name):
    """
    Runs a completely random policy on the environment
    :param run_name: (str) name for logging files
    :return:
    """
    hw = 15
    action_force = 0.6
    scaling = 2
    res = 32
    env = envs.BillardsEnv(n=3, hw=15, r=1., res=res)
    task = task(env, action_force=0.6, num_stacked=8)
    # env = envs.BillardsEnv(n=3, hw=hw, r=1.)
    # task = task(env, action_force=action_force)
    task.scaling = scaling
    
    all_rewards = []

    for t in tqdm(range(100)):
        imgs = []
        for i in tqdm(range(100)):
            # mcts = MCTS(task, max_rollout=40)
            # action = mcts.run_mcts(300)
            action = np.random.randint(9)
            img, _, r, _ = task.step(action)
            all_rewards.append(r)

    return all_rewards
