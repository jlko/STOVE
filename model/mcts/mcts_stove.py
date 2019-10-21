import sys

import model.envs as envs
import numpy as np
import torch
import time
import imageio
import copy
from tqdm import tqdm

from multiprocess import Pool
import time


def multi_one_hot(actions, action_shape):
    """
    transforms a list of actions into a list of one hot encoded vectors
    :param actions: (list int) the list of action indices
    :param action_shape: (int) the maximum number of entries in the one hot
    vector
    :return: the created sequence
    """
    l = len(actions)
    full_vec = torch.zeros((l, action_shape))
    full_vec[range(l), actions] = 1.
    return full_vec.unsqueeze(0)


def tile(a, dim, n_tile):
    """
    Recreates the torch repeat_interleaved function
    :param a: a vector that should be tiled
    :param dim: the dimension along which the vector should be tiled
    :param n_tile: the number of repeats per tile
    :return: the tiled representation
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        'cuda')
    return torch.index_select(a.to('cuda'), dim, order_index)


def encode_img(img):
    """
    Encodes a numpy array into a torch Tensor and permutes channels
    :param img: (batch, batch, width, height, channels)) the numpy array
    representing an image
    :return: (batch, batch, channels width, height) a torch Tensor
    """
    img_torch = torch.Tensor(img)
    img_torch = img_torch.permute(0, 1, 4, 2, 3)
    # infer initial model state
    return img_torch


def select(mcts):
    """
    Rephrases mcts.select as a top level function for parallelization,
    since it is side effect free
    """
    return mcts.select('r', mcts.Zstate['r'])


class BatchedMCTSHandler():
    """
    Class which holds and runs a batched MCTS tree instances
    """

    def __init__(self, trees, appearances, action_space=9, max_rollout_depth=20):
        """
        Initializes all trees,
        :param trees: (list MCTS) a list of MCTS tree instances
        :param appearances: the object appearance vector from STOVE
        :param action_space: the length of the action space
        :param max_rollout: the maximum rollout depth
        """
        self.num_mcts = len(trees)
        self.trees = trees
        self.max_rollout = max_rollout_depth
        self.obj_app = appearances
        self.actions = action_space

    def run_mcts(self, env, runs_per_round):
        """
        Runs all batched MCTS instances concurrently on the STOVE model
        :param env: (STOVE) a STOVE instance representing the env
        :param runs_per_round: (int) the number of MCTS expansions to perform
        :return: an array of next actions
        """
        pool = Pool(self.num_mcts)
        for i in range(runs_per_round):
            start = time.time()
            result = pool.imap(select, self.trees)
            all_states = []
            all_zs = []
            for state, z in result:
                all_states.append(state)
                all_zs.append(z)
            # expand all mcts by applying all next actions batched on all mcts zs
            expansion_actions = multi_one_hot(range(self.actions),
                                              self.actions)
            expansion_actions = expansion_actions.view(self.actions, 1,
                                                       self.actions)
            expansion_actions = expansion_actions.repeat(self.num_mcts, 1,
                                                         1).to('cuda')
            new_zs, r = env.rollout(
                tile(torch.cat(all_zs, 0), 0, self.actions).to('cuda'),
                num=1,
                actions=expansion_actions,
                appearance=tile(self.obj_app, 0, self.actions).to('cuda')
            )

            # rollout all new expanded nodes in parallel
            random_rollout_actions = np.random.randint(
                self.actions,
                size=(self.actions * self.num_mcts * self.max_rollout * 2,))
            random_rollout_actions = multi_one_hot(
                random_rollout_actions, self.actions)
            random_rollout_actions = random_rollout_actions.view(
                self.num_mcts * self.actions, self.max_rollout * 2,
                self.actions)
            _, r_rollout = env.rollout(
                new_zs[:, -1].to('cuda'),
                num=self.max_rollout * 2,
                actions=random_rollout_actions,
                appearance=tile(self.obj_app, 0, self.actions).to('cuda')
            )

            for j, mcts in enumerate(self.trees):
                low = j * self.actions
                high = (j + 1) * self.actions
                mcts.backpropagate(new_zs[low:high], r[low:high],
                                   r_rollout[low:high], all_states[j])

        pool.close()
        actions = []
        for i in range(self.num_mcts):
            counts = [self.trees[i].Nsa['r' + str(a)] for a in
                      range(self.actions)]
            actions.append(np.argmax(counts))
        return actions


class MCTS():
    def __init__(self, appearance, inferred_z, action_space=9, max_rollout_depth=20):
        # self.children = {}
        self.actions = action_space
        self.Nsa = {'r': 0}
        self.Qsa = {'r': 0}
        self.Ns = {'r': 0}
        self.Zstate = {'r': inferred_z.cpu()}
        self.c = 1.
        self.app = appearance.cpu()
        self.max_rollout = max_rollout_depth

    def select(self, s, z):

        def uct(q, n, N):
            return q + self.c * np.sqrt(np.log(N) / (1 + n))

        cur_best = -float('inf')
        best_act = 0

        final = False
        while not final:
            if not (s + str(0) in self.Qsa.keys()) or len(
                    s) == self.max_rollout:
                final = True
                z = self.Zstate[s]
            else:
                for next_a in range(self.actions):
                    u = uct(self.Qsa[s + str(next_a)],
                            self.Nsa[s + str(next_a)], self.Ns[s])
                    if u > cur_best:
                        cur_best = u
                        best_act = next_a
                # don't step, get next z from dict
                s = s + str(best_act)
        return s, z

    def backpropagate(self, new_zs, rs, r_rollout, s):
        rs = rs - 1
        r_rollout = r_rollout - 1
        remaining = self.max_rollout * 2 - len(s) + 1
        assert remaining >= 0
        discounted_action_values = 0
        for a in range(self.actions):
            sa = s + str(a)
            discounted = (rs[a] * (0.95 ** len(s)) + torch.sum(
                r_rollout[a, :remaining] * 0.95 ** torch.Tensor(
                    range(len(s), self.max_rollout)))).item()
            discounted_action_values += discounted
            self.Qsa[sa] = discounted
            self.Nsa[sa] = 1
            self.Zstate[s + str(a)] = new_zs[a].cpu()
            self.Ns[s + str(a)] = 0
        self.Ns[s] += 1

        backprop_value = discounted_action_values / self.actions

        backprop_states = s
        for index in range(1, len(s)):
            update = backprop_states
            s = backprop_states[:-1]
            self.Ns[s] += 1
            self.Nsa[update] += 1
            self.Qsa[update] = (self.Qsa[update] * (
                        self.Nsa[update] - 1) + backprop_value) / self.Nsa[
                                   update]
            backprop_states = s


def initialize_img(envs, steps=8, res=32):
    num_parallel_envs = len(envs)
    img = np.zeros((num_parallel_envs, steps, res, res, 3))
    for i in range(num_parallel_envs):
        for j in range(steps):
            ret_img, _, _, _ = envs[i].step(0)
            img[i, j] = ret_img
    actions = multi_one_hot([0] * steps * num_parallel_envs, 9)
    actions = actions.view(num_parallel_envs, steps, 9)
    return img, actions


def update_buffer(img, new_img, action, new_action):
    img[:, :-1] = img[:, 1:]
    img[:, -1] = new_img
    new_action = multi_one_hot(new_action, 9)
    action[:, :-1] = action[:, 1:]
    action[:, -1] = new_action
    return img, action


def run_mcts_model(img, model, actions, num_parallel_envs=100,
                   mcts_steps=100, max_rollout_depth=10):
    """
    Wrapper function to run the STOVE based MCTS
    :param img: (batch, batch, width, height, channels) the current (real) env state
    :param model: (STOVE) the STOVE instance to use
    :param actions: the last taken actions (needed to initialize STOVE)
    :param num_parallel_envs: the number of envs to run in parallel
    :param mcts_steps: the number of nodes to exand in each tree
    :param max_rollout: the maximum rollout depth per tree
    :return: the list of next actions
    """
    with torch.no_grad():
        _, prop_dict, _ = model(encode_img(img), 0, actions=actions,
                                pretrain=False)
        all_mcts = [MCTS(prop_dict['obj_appearances'][env:env + 1, -1],
                         prop_dict['z'][env:env + 1, -1],
                         max_rollout_depth=max_rollout_depth) for env in
                    range(num_parallel_envs)]
        mcts = BatchedMCTSHandler(
            all_mcts,
            prop_dict['obj_appearances'][:, -1],
            action_space=9,
            max_rollout_depth=max_rollout_depth)
        all_actions = mcts.run_mcts(model, mcts_steps)
    return all_actions
