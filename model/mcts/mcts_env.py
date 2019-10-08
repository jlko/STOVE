import sys
import os

import envs
import numpy as np
import time 
import imageio
import copy
from tqdm import tqdm

class MCTS():
    def __init__(self, env, max_rollout=20):
        self.root_x = np.array(env.env.x, copy=True)
        self.root_v = np.array(env.env.v, copy=True)
        self.env = env
        # self.children = {}
        self.actions = env.get_action_space()
        self.Nsa = {'r': 0}
        self.Qsa = {'r': 0}
        self.Ns = {'r': 0}
        self.c = 1.
        self.max_rollout = max_rollout

    def run_mcts(self, runs_per_round):
        for i in range(runs_per_round):
            self.select(self.env, 'r')
            self.env_reset()
        counts = [self.Nsa[('r', a)] for a in range(self.actions)]
        # print("counts ", counts)
        # print("Q-values", [self.Qsa[('r', a)] for a in range(self.actions)])
        # print()
        return np.argmax(counts)
    
    def select(self, env, s):

        def uct(q, n, N):
            return q + self.c * np.sqrt(np.log(N)/(1 + n))

        cur_best = -float('inf')
        best_act = 0
        
        expanded = False

        if not ((s, 0) in self.Qsa.keys()) and len(s) != self.max_rollout:
            best_act, backprop_value = self.expand(s, env)
            expanded = True
        elif len(s)+1 != self.max_rollout:
            for next_a in range(self.actions):
                u = uct(self.Qsa[(s, next_a)], self.Nsa[(s, next_a)], self.Ns[s])
                if u > cur_best:
                    cur_best = u
                    best_act = next_a
            env.step(best_act)
            backprop_value = self.select(env, s + str(best_act))
            expanded = True
        else:
            best_action = np.random.randint(self.actions)
            backprop_value = self.Qsa[(s, best_action)]
        
        self.Ns[s] += 1
        self.Nsa[(s, best_act)] += 1
        if expanded:
            self.Qsa[(s, best_act)] = (self.Qsa[(s, best_act)] * (self.Nsa[(s, best_act)] - 1) + backprop_value) / self.Nsa[(s, best_act)]
        
        return backprop_value

    def expand(self, s, env):
        self.Ns[s] = 0
        for a in range(self.actions):
            sa = (s, a)
            self.Qsa[sa] = -1
            self.Nsa[sa] = 0
        next_action = np.random.randint(self.actions)
        _, _, r, _ = env.step(next_action)
        rollout_r = self.rollout(env, len(s))
        total_r = r + rollout_r

        return next_action, total_r

    def rollout(self, env, cur_depth):
        total_r = 0
        for i in range(self.max_rollout - cur_depth):
            next_act = np.random.randint(self.actions)
            _, _, r, _ = env.step(next_act)
            total_r += r
        return total_r

    def env_reset(self):
        self.env.env.x = np.array(self.root_x, copy=True)
        self.env.env.v = np.array(self.root_v, copy=True)


def main(task, run_name):
    hw = 15
    action_force = 0.6
    scaling = 2
    task.scaling = scaling
    results = []
    for t in tqdm(range(100)):
        env = envs.BillardsEnv(n=3, hw=hw, r=1.)
        env_task = task(env, action_force=action_force)
        imgs = []
        for i in tqdm(range(100)):
            mcts = MCTS(env_task, max_rollout=10)
            action = mcts.run_mcts(100)
            img, _, r, _ = env_task.step(action)
            results.append(r)
            imgs.append(img)
        imgs = (255 * np.array(imgs)).astype(np.uint8)
        path = os.getcwd() + '/{}_{}_{}_{}'.format(type(task).__name__, action_force, hw, scaling).replace('.', '-')

        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
        imageio.mimsave(path + '/{}_{}.gif'.format(run_name, t), imgs, fps=24)
    print()
    print(np.mean(results))


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
