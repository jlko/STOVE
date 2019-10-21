import sys
import os

import model.envs as envs
import numpy as np
import time
import imageio
import copy
from tqdm import tqdm


class MCTS():
    """
    Main MCTS wrapper class containing tree and expansion policy
    """

    def __init__(self, env, max_rollout=20):
        """
        Initializes all tree components as dictionaries for fast lookup
        :param env: the environment the MCTS is running on (will be reset)
        :param max_rollout: the maximum rollout depth (needed for endless tasks)
        """
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
        """
        The main MCTS call method. Expands the tree and returns the next best
        action based on node visiting frequency
        :param runs_per_round: (int) number of MCTS nodes to expand
        :return: the most visited root child node index (corresponding to best
        next action)
        """
        for i in range(runs_per_round):
            self.select(self.env, 'r')
            self.env_reset()
        counts = [self.Nsa[('r', a)] for a in range(self.actions)]
        # print("counts ", counts)
        # print("Q-values", [self.Qsa[('r', a)] for a in range(self.actions)])
        # print()
        return np.argmax(counts)

    def select(self, env, s):
        """
        Selects the next node to expand, expands it and propagates the value
        back along the tree
        :param env: the env that represents the current game state
        :param s: the current node
        :return: the backpropagated value
        """

        def uct(q, n, N):
            return q + self.c * np.sqrt(np.log(N) / (1 + n))

        cur_best = -float('inf')
        best_act = 0

        expanded = False

        if not ((s, 0) in self.Qsa.keys()) and len(s) != self.max_rollout:
            best_act, backprop_value = self.expand(s, env)
            expanded = True
        elif len(s) + 1 != self.max_rollout:
            for next_a in range(self.actions):
                u = uct(self.Qsa[(s, next_a)], self.Nsa[(s, next_a)],
                        self.Ns[s])
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
            self.Qsa[(s, best_act)] = (self.Qsa[(s, best_act)] * (
                    self.Nsa[(s, best_act)] - 1) + backprop_value) / \
                                      self.Nsa[(s, best_act)]

        return backprop_value

    def expand(self, s, env):
        """
        Expands a leaf node based on the env
        :param s: the leaf node
        :param env: the environment representing the game state
        :return: the expanded value and the rollout result
        """
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
        """
        Rolls out the environment along a random policy
        :param env: the environment representing the game state
        :param cur_depth: the current depth of the already expanded tree
        :return: the total result of the rollout
        """
        total_r = 0
        for i in range(self.max_rollout - cur_depth):
            next_act = np.random.randint(self.actions)
            _, _, r, _ = env.step(next_act)
            total_r += r
        return total_r

    def env_reset(self):
        """
        Reset the env
        """
        self.env.env.x = np.array(self.root_x, copy=True)
        self.env.env.v = np.array(self.root_v, copy=True)
