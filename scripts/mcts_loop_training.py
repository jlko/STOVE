import numpy as np
import torch
import pickle
import os
from tqdm import tqdm

from model.mcts.mcts_stove import run_mcts_model, initialize_img, \
    update_buffer, encode_img
from model.video_prediction.config import StoveConfig as Config
from model.video_prediction.stove import Stove
from model.video_prediction.train import Trainer
from model.video_prediction.load_data import load, StoveDataset
from model.envs import envs
from model.main import main, config_setup
from model.spn.probabilistic_models import _get_bg_spn, _get_obj_spn


def mcts_based_sampling(model, run_len=100, num_parallel_envs=10,
                        depth=10, mcts_steps=20, res=32, n=3):
    config = {
        'res': 32, 'hw': 10, 'n': 3, 't': 1., 'm': 1.,  # dt = 0.81
        'granularity': 50, 'r': 1}
    sars = []
    all_envs = [
        envs.AvoidanceTask(envs.BillardsEnv(n=n, hw=10, r=1., res=res, seed=s),
                           action_force=0.6, num_stacked=8) for s in
        range(num_parallel_envs)]

    img, actions = initialize_img(all_envs)

    all_imgs = np.zeros((num_parallel_envs, run_len, res, res, 3))
    all_states = np.zeros((num_parallel_envs, run_len, n, 4))
    all_actions = np.zeros((num_parallel_envs, run_len, 9))
    all_rewards = np.zeros((num_parallel_envs, run_len, 1))
    all_dones = np.zeros((num_parallel_envs, run_len, 1))

    for i in tqdm(range(run_len)):
        mcts_actions = run_mcts_model(img,
                                      model,
                                      actions,
                                      max_rollout_depth=depth,
                                      num_parallel_envs=num_parallel_envs,
                                      mcts_steps=mcts_steps)
        for j in range(num_parallel_envs):
            ret_img, state, r, done = all_envs[j].step(mcts_actions[j])
            all_imgs[j, i] = ret_img
            all_states[j, i] = state
            action = np.zeros(9)
            action[mcts_actions[j]] = 1.
            all_actions[j, i - 1] = action
            all_rewards[j, i] = r
            all_dones[j, i] = done
        img, action = update_buffer(img, all_imgs[:, i], actions, mcts_actions)
    data = dict()
    data['X'] = all_imgs
    data['y'] = all_states
    data['action'] = all_actions
    data['reward'] = all_rewards
    data['done'] = all_dones
    data['type'] = 'avoidance'
    data['action_force'] = 0.6

    data.update({'action_space': 9})
    data.update(config)
    data['coord_lim'] = config['hw']

    return data


def update_batch(old_batch, new_batch, update_ratio=0.5):
    config = {
        'res': 32, 'hw': 10, 'n': 3, 't': 1., 'm': 1.,  # dt = 0.81
        'granularity': 50, 'r': 1}
    data = dict()
    data['X'] = np.zeros_like(old_batch['X'])
    data['y'] = np.zeros_like(old_batch['y'])
    data['action'] = np.zeros_like(old_batch['action'])
    data['reward'] = np.zeros_like(old_batch['reward'])
    data['done'] = np.zeros_like(old_batch['done'])
    data['type'] = 'avoidance'
    data['action_force'] = 0.6

    num_new = int(len(old_batch['X']) * update_ratio)
    num_keep = len(old_batch['X']) - num_new
    keep = np.random.choice(len(old_batch['X']), num_keep, replace=False)
    new = np.random.choice(len(new_batch['X']), num_new, replace=False)

    data['X'][:num_keep] = old_batch['X'][keep]
    data['y'][:num_keep] = old_batch['y'][keep]
    data['action'][:num_keep] = old_batch['action'][keep]
    data['reward'][:num_keep] = old_batch['reward'][keep]
    data['done'][:num_keep] = old_batch['done'][keep]

    data['X'][num_keep:] = new_batch['X'][new]
    data['y'][num_keep:] = new_batch['y'][new]
    data['action'][num_keep:] = new_batch['action'][new]
    data['reward'][num_keep:] = new_batch['reward'][new]
    data['done'][num_keep:] = new_batch['done'][new]

    data['type'] = 'avoidance'
    data['action_force'] = 0.6

    data.update({'action_space': 9})
    data.update(config)
    data['coord_lim'] = config['hw']

    return data


def save_data(save, data, run_types):
    path = save.format(run_types) + '.pkl'
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    return path


def train_loop(run_name='default',
               total_samples=1000000,
               epochs=[50] * 10,
               batch_size=256,
               parallel_envs=100,
               mcts_steps=100,
               depth=10,
               base_path='.',
               train_data_loc='avoidance_train.pkl',
               test_data_loc='avoidance_test.pkl'):

    epoch_samples = int(total_samples/len(epochs))
    run_len = int(epoch_samples / parallel_envs)

    print('''
==============================================================
    Starting training full loop mcts with the following data:
    {} total samples over {} runs
    {} parallel envs in a mcts run len of {}
==============================================================
    '''.format(total_samples, len(epochs), parallel_envs, run_len))
    
    base = f'{base_path}/experiments/mcts_loop/{run_name}/'
    
    extras = {'experiment_dir': base,
              'traindata': f'{base_path}/data/{train_data_loc}',
              'testdata': f'{base_path}/data/{test_data_loc}',
              'debug_core_appearance': True,
              'debug_match_appearance': False,
              'num_epochs': epochs[0],
              'batch_size': batch_size}
    trainer = main(extras=extras)

    if not os.path.isdir(base):
        os.mkdir(base + '/stove_logging/')
        os.mkdir(base + '/samples/')

    batch = pickle.load(open(extras['traindata'], 'rb'))
    update_batch(batch, batch, )
    
    total_epochs = 0
    total_steps = 0

    for run_cycle, epoch in enumerate(epochs):
        save_data(base + 'samples/batch_{}', batch, run_cycle)
        total_epochs += epoch
        total_steps += epoch * len(batch['X'])
        train_dataset = StoveDataset(trainer.c, data=batch)
        trainer.dataloader = train_dataset
        training_results = trainer.train(epoch)
        save_data(base + 'stove_logging/stove_{}', training_results, run_cycle)
        trainer.save(total_epochs, total_steps)

        with torch.no_grad():
            new_data = mcts_based_sampling(trainer.stove,
                    run_len=run_len, 
                    num_parallel_envs=parallel_envs,
                    mcts_steps=mcts_steps,
                    depth=depth)
        save_data(base + 'samples/samples_{}', new_data, run_cycle)
        batch = update_batch(batch, data, update_ratio = len(new_data['X'])/len(batch['X']))
