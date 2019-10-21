import sys
import os
import pickle

import numpy as np
import time
import imageio
import copy
from tqdm import tqdm
import torch

from model.envs import envs
from model.mcts.mcts_env import MCTS
from model.mcts.mcts_stove import run_mcts_model, initialize_img, update_buffer
from model.main import main


def main_mcts_model(run_name, restore_point,
                    save_gifs=True,
                    num_parallel_envs=100,
                    mcts_steps=100,
                    max_rollout_depth=10,
                    run_len=100):
    """
    Runs the MCTS agent on a pretrained STIVE world model
    :param run_name: (str) name for logging files
    :param restore_point: (str) file location of the STOVE model
    :param train_data_loc: (str) file location of the STOVE training data for reference
    :param test_data_loc: (str) file location of the STOVE testing data for reference
    :param save_gifs: (bool) flag enabling the logging of the gifs
    :return: total results from the run
    """
    extras = {'nolog': True, 'traindata': './data/avoidance_train.pkl',
              'testdata': './data/avoidance_test.pkl'}
    trainer = main(extras=extras, restore=restore_point)

    model = trainer.stove

    with torch.no_grad():
        res = 32
        # env = envs.BillardsEnv(n=3, hw=10, r=1., res=res)
        # task = envs.AvoidanceTask(env, action_force=0.6, num_stacked=8)

        for t in tqdm(range(1)):
            all_envs = [envs.AvoidanceTask(
                envs.BillardsEnv(n=3, hw=10, r=1., res=res, seed=s),
                action_force=0.6, num_stacked=8) for s in
                range(num_parallel_envs)]

            img, actions = initialize_img(all_envs)
            results = []
            # initialize frame_buffer in env
            # infer initial model state

            imgs = []
            for i in range(num_parallel_envs):
                imgs.append([])

            results = []
            for i in tqdm(range(run_len)):
                next_actions = run_mcts_model(img,
                                              model,
                                              actions,
                                              max_rollout_depth=max_rollout_depth,
                                              num_parallel_envs=num_parallel_envs,
                                              mcts_steps=mcts_steps)
                for j in range(num_parallel_envs):
                    ret_img, _, r, _ = all_envs[j].step(next_actions[j])
                    imgs[j].append(ret_img)
                    results.append(float(r))
                img, actions = update_buffer(img, ret_img, actions,
                                             next_actions)
            imgs = np.array(imgs)

            if save_gifs:
                for i in range(num_parallel_envs):
                    print_imgs = (255 * imgs[i]).astype(np.uint8)
                    imageio.mimsave(
                        './{}_{}_{}.gif'.format(run_name, t, i),
                        print_imgs, fps=24)
            pickle.dump(results, open('quicksave', 'wb'))
            return results


def main_mcts_env(run_name,
                    save_gifs=True,
                    num_parallel_envs=100,
                    mcts_steps=100,
                    max_rollout_depth=10,
                    run_len=100):
    hw = 15
    action_force = 0.6
    scaling = 2
    results = []
    for t in tqdm(range(num_parallel_envs)):
        env_task = envs.AvoidanceTask(env, action_force=action_force)
        imgs = []
        for i in tqdm(range(run_len)):
            mcts = MCTS(env_task, max_rollout=max_rollout_depth)
            action = mcts.run_mcts(mcts_steps)
            img, _, r, _ = env_task.step(action)
            results.append(r)
            imgs.append(img)
        if save_gifs:
            imgs = (255 * np.array(imgs)).astype(np.uint8)
            path = os.getcwd() + '/mcts_results/'

            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            imageio.mimsave(path + '/{}_{}.gif'.format(run_name, t), imgs,
                            fps=24)

    return results
