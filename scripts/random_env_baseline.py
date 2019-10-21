import sys
import os

import envs
import numpy as np
import time 
import imageio
import copy
from tqdm import tqdm

def main(task, run_name):
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

    print(np.mean(all_rewards))
    print(np.std(all_rewards))


if __name__ == "__main__":
    run_name = sys.argv[1]
    task = sys.argv[2]
    if task == 'avoidance':
        task = envs.AvoidanceTask
    elif task == 'mindist':
        task = envs.MinDistanceTask
    elif task == 'maxdist':
        task = envs.MaxDistanceTask
    else:
        raise ValueError('task is unkown')
    main(task, run_name)
