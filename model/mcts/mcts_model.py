import sys

import envs
import numpy as np
import torch
import time
import imageio
import copy
from tqdm import tqdm

from vin import main

from multiprocess import Pool
import time


def multi_one_hot(actions, action_shape):
    l = len(actions)
    full_vec = torch.zeros((l, action_shape))
    full_vec[range(l), actions] = 1.
    return full_vec.unsqueeze(0)

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to('cuda')
    return torch.index_select(a.to('cuda'), dim, order_index)

def select(mcts):
    return mcts.select('r', mcts.Zstate['r'])

def encode_img(img):
    img_torch = torch.Tensor(img)
    img_torch = img_torch.permute(0, 1, 4, 2, 3)
    # infer initial model state
    return img_torch


class BatchedMCTSHandler():
    def __init__(self, trees, appearances, action_space=9, max_rollout=20):
        self.num_mcts = len(trees)
        self.trees = trees
        self.max_rollout = max_rollout
        self.obj_app = appearances
        self.actions = action_space

    def run_mcts(self, env, runs_per_round):
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
            expansion_actions = multi_one_hot(range(self.actions), self.actions)
            expansion_actions = expansion_actions.view(self.actions, 1, self.actions)
            expansion_actions = expansion_actions.repeat(self.num_mcts, 1, 1).to('cuda')
            new_zs, r = env.rollout(
                    torch.cat(all_zs, 0).repeat_interleave(self.actions, 0).to('cuda'),
                    num=1,
                    actions=expansion_actions,
                    appearance=self.obj_app.repeat_interleave(self.actions, 0).to('cuda')
                    )

            # rollout all new expanded nodes in parallel
            random_rollout_actions = np.random.randint(
                    self.actions, size=(self.actions * self.num_mcts * self.max_rollout * 2,))
            random_rollout_actions = multi_one_hot(
                    random_rollout_actions, self.actions)
            random_rollout_actions = random_rollout_actions.view(
                    self.num_mcts * self.actions, self.max_rollout * 2, self.actions)
            _, r_rollout = env.rollout(
                    new_zs[:, -1].to('cuda'),
                    num=self.max_rollout * 2, 
                    actions=random_rollout_actions, 
                    appearance=self.obj_app.repeat_interleave(self.actions, 0).to('cuda')
                    )

            for j, mcts in enumerate(self.trees):
                low = j * self.actions
                high = (j+1) * self.actions
                mcts.backpropagate(new_zs[low:high], r[low:high], r_rollout[low:high], all_states[j])

        pool.close()
        actions = []
        for i in range(self.num_mcts):
            counts = [self.trees[i].Nsa['r' + str(a)] for a in range(self.actions)]
            actions.append(np.argmax(counts))
        return actions


class MCTS():
    def __init__(self, appearance, inferred_z, action_space=9, max_rollout=20):
        # self.children = {}
        self.actions = action_space
        self.Nsa = {'r': 0}
        self.Qsa = {'r': 0}
        self.Ns = {'r': 0}
        self.Zstate = {'r': inferred_z.cpu()}
        self.c = 1.
        self.app = appearance.cpu()
        self.max_rollout = max_rollout

    def select(self, s, z):

        def uct(q, n, N):
            return q + self.c * np.sqrt(np.log(N)/(1 + n))

        cur_best = -float('inf')
        best_act = 0

        final = False
        while not final:
            if not (s + str(0) in self.Qsa.keys()) or len(s) == self.max_rollout:
                final = True
                z = self.Zstate[s]
            else:
                for next_a in range(self.actions):
                    u = uct(self.Qsa[s + str(next_a)], self.Nsa[s + str(next_a)], self.Ns[s])
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
            discounted =  (rs[a] * (0.95 ** len(s))  + torch.sum(r_rollout[a, :remaining] * 0.95 ** torch.Tensor(range(len(s), self.max_rollout)))).item()
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
            self.Qsa[update] = (self.Qsa[update] * (self.Nsa[update] - 1) + backprop_value) / self.Nsa[update]
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
    img[j, :-1] = img[j, 1:]
    img[j, -1] = new_img
    new_action = multi_one_hot([new_action], 9)
    action[j, :-1] = action[j, 1:]
    action[j, -1:] = new_action
    return img, action


def run_mcts_model(img, model, rollouts=10, num_parallel_envs=100, mcts_steps=100):
    with torch.no_grad():
        _, prop_dict, _ = model(encode_img(img), 0, action=actions, pretrain=False)
        all_mcts = [MCTS(prop_dict['obj_appearances'][env:env+1, -1],
            prop_dict['z'][env:env+1, -1],
                max_rollout=rollouts) for env in range(num_parallel_envs)]
        mcts = BatchedMCTSHandler(
                all_mcts,
                prop_dict['obj_appearances'][:, -1],
                action_space=9,
                max_rollout=max_rollout)
        all_actions = mcts.run_mcts(model, mcts_steps)
    return all_actions


def main_mcts(task, run_name):
    user = 'user'
    restore = '/home/{}/share/good_rl_run'.format(user)
    extras = {
        'nolog': True, 'traindata': '/home/{}/share/data/billards_w_actions_train_data.pkl'.format(user),
        'testdata': '/home/{}/share/data/billards_w_actions_test_data.pkl'.format(user)}
    trainer = main(extras=extras, restore=restore)

    model = trainer.net

    with torch.no_grad():
        res = 32
        # env = envs.BillardsEnv(n=3, hw=10, r=1., res=res)
        # task = task(env, action_force=0.6, num_stacked=8)

        num_parallel_envs = 100
        for t in tqdm(range(1)):
            all_envs = [envs.AvoidanceTask(envs.BillardsEnv(n=3, hw=10, r=1., res=res, seed=s), action_force = 0.6, num_stacked=8) for s in range(num_parallel_envs)]

            img, actions = initialize_img(all_envs)
            results = []
            # initialize frame_buffer in env
            # infer initial model state

            imgs = []
            for i in range(num_parallel_envs):
                imgs.append([])

            results = []
            for i in tqdm(range(100)):
                run_mcts_model(img, 
                        model, 
                        rollouts=10, 
                        num_parallel_envs=100, 
                        mcts_steps=100)
                for j in range(num_parallel_envs):
                    ret_img, _, r, _ = all_envs[j].step(all_actions[j])
                    imgs[j].append(ret_img)
                    results.append(float(r))
                    update_buffer(img, ret_img, actions, all_actions[j])
                # TODO: this seems inefficient. We should reuse the model state from before, instead of inferring it again
            print()
            print()
            print(np.mean(np.array(results)))
            imgs = np.array(imgs)
            for i in range(num_parallel_envs):
                print_imgs = (255 * imgs[i]).astype(np.uint8)
                imageio.mimsave('./{}_{}_{}_{}.gif'.format(run_name, type(task).__name__, t, i), print_imgs, fps=24)


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
    main_mcts(task, run_name)
