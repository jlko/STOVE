from mcts_model import encode_img, initialize_img, run_mcts_model
import train
from config import Config
import envs


def mcts_based_sampling(model, config, run_len=100, num_parallel_envs=100, depth=10, mcts_runs=50, res=32, n=3):
    sars = []
    all_envs = [envs.AvoidanceTask(envs.BillardsEnv(n=n, hw=10, r=1., res=res, seed=s), action_force = 0.6, num_stacked=8) for s in range(num_parallel_envs)]

    img, actions = initialize_img(all_envs)
    
    all_imgs = np.zeros((num_parallel_envs, run_len, res, res, 3))
    all_states = np.zeros((num_parallel_envs, run_len, n, 4))
    all_actions = np.zeros((num_parallel_envs, run_len, 9))
    all_rewards = np.zeros((num_parallel_envs, run_len, 1))
    all_dones = np.zeros((num_parallel_envs, run_len, 1))
    
    for i in tqdm(range(run)):
        run_mcts_model(img, 
                model, 
                rollouts=depth, 
                num_parallel_envs=num_parallel_envs, 
                mcts_steps=mcts_runs)
        for j in range(num_parallel_envs):
            ret_img, state, r, done = all_envs[j].step(all_actions[j])
            sars.append((img, all_actions[j], r, ret_img))
            all_images[j, i] = ret_img
            all_states[j, i] = state
            action = np.zeros(9)
            action[all_actions[j]] = 1.
            all_actions[j, i-1] = action
            all_rewards[j, i] = r
            all_dones[j, i] = done
            img, action = update_buffer(img, ret_img, actions, all_actions[j])
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
    data['X'] = np.zeros_like(old_batch['X'])
    data['y'] = np.zeros_like(old_batch['Y'])
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
    data['Y'][:num_keep] = old_batch['X'][keep]
    data['action'][:num_keep] = old_batch['action'][keep]
    data['reward'][:num_keep] = old_batch['reward'][keep]
    data['done'][:num_keep] = old_batch['done'][keep]

    data['X'][num_keep:] = new_batch['X'][new]
    data['Y'][num_keep:] = new_batch['X'][new]
    data['action'][num_keep:] = new_batch['action'][new]
    data['reward'][num_keep:] = new_batch['reward'][new]
    data['done'][num_keep:] = new_batch['done'][new]
    
    return data


def save_data(save, data, run_types):
    path = save.format(run_types) + '.pkl'
    f = open(path, "wb")
    pickle.dump(data, f, protocol=4)
    f.close()
    return path


def main(total_sample=100000, epoch_samples=10000, epochs=[40,40,30,30,20,20,10,10,10,10], batch_size=256, parallel_envs=100, mcts_runs=50, depth=10):
    run_cycles = int(total_sample/epoch_samples)
    assert len(epochs) == run_cycles

    run_len = int(epoch_samples/parallel_envs)

    print('''
==============================================================
    Starting training full loop mcts with the following data:
    {} total samples over {} runs
    {} parallel envs in a mcts run len of {}
==============================================================
    '''.format(total_sample, run_cycles, parallel_envs, run_len))

    user = 'user'
    restore = '/home/{}/share/good_rl_run'.format(user)
    extras = {'nolog': True,
            'traindata': '/home/user/share/data/billards_w_actions_train_data.pkl'.format(user), 
            'testdata': '/home/user/share/data/billards_w_actions_test_data.pkl'.format(user)}
    trainer = main(extras=extras, restore=restore)

    config = trainer.c
    net = Net(config)
    net = net.to(config.device)
    net = net.type(config.dtype)

    initial_data_sample = 
    initial_test_sample = trainer.testset
    
    all_test_samples = [initial_data_sample]

    batch = initial_data_sample

    for run_cycle in range(run_cycles):
        trainer = Trainer(net, config, batch, initial_test_sample)
        trainer.train()
        
        with torch.no_grad():
            data = mcts_based_sampling(trainer.net, config, run_len, parallel_envs)
        save('./data/mcts_loop/samples_{}', data, run_cylcle)
        batch = update_batch(batch, data)
        path = save('./data/mcts_loop/batch_{}', batch, run_cylcle)
        config.update({'traindata': path})
    
if __name__ == "__main__":
    main()
