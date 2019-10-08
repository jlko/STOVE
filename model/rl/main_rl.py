import numpy as np
import torch
import argparse
import json
import os
import traceback
from tqdm import tqdm
import gym
import pickle

from tensorboardX import SummaryWriter

# TODO: this is ugly as hell but python sucks sometimes, should try to put everything in packages?
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from rl_model import Policy
from agents import PPOAgent
from runners import PPORunner
from envs import BillardsEnv, AvoidanceTask, MaxDistanceTask, MinDistanceTask
from vin import main as load_vin

def parse_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
            'experiment_id', 
            help='name of the experiment')
    parser.add_argument(
            '--env', default='billards', 
            help='environment to use: billards')
    parser.add_argument(
            '--task', default='avoidance', 
            help='task to learn if training: avoidance | maxdist | mindist')
    parser.add_argument(
            '--seed', type=int, default=0, 
            help='random seed')
    parser.add_argument(
            '--lr', type=float, default=2e-4, 
            help='learning rate (default: 1e-4)')
    parser.add_argument(
            '--clip-param', type=float, default=0.1,
            help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
            '--num-steps', type=int, default=128,
            help='number of forward steps in PPO (default: 128)')   
    parser.add_argument(
            '--batch-size', type=int, default=32,
            help='number of trajectories sampled per batch (default: 32)')          
    parser.add_argument(
            '--num-env-steps', type=int, default=10000000,
            help='number of total environment steps (default: 10000000)')
    parser.add_argument(
            '--num-ppo-mb', type=int, default=32,
            help='number of batches for ppo (default: 32)')     
    parser.add_argument(
            '--num-ppo-epochs', type=int, default=4,
            help='number of epochs for ppo (default: 4)')      
    parser.add_argument(
            '--save-interval', type=int, default=100,
            help='save interval, one save per n batches (default: 100)')
    parser.add_argument(
            '--value-loss-coef', type=float, default=0.5,
            help='value loss coefficient (default: 0.5)')
    parser.add_argument(
            '--entropy-coef', type=float, default=0.01,
            help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
            '--eps', type=float, default=1e-5,
            help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument(
            '--max-grad-norm', type=float, default=40,
            help='max norm of gradients (default: 40)')
    parser.add_argument(
            '--summary-dir', type=str, default='./summary/',
            help='directory to save agent tb summaries and args (default: ./summary/)')
    parser.add_argument(
            '--model-dir', type=str, default='./models/', 
            help='directory to save models (default: ./models/)')
    parser.add_argument(
            '--hidden-size', type=int, default=512,
            help='Hidden-size in policy head')
    parser.add_argument(
            '--gym', action='store_true', default=False,
            help='whether or not to use gym as environment')
    parser.add_argument(
            '--use-states', action='store_true', default=False,
            help='use states instead of images when true')
    parser.add_argument(
            '--use-deep-layers', action='store_true', default=False,
            help='use more linear layers in head')

    parser.add_argument(
            '--num-balls', type=int, default=3,
            help='Number of balls to simulate in the environment (default: 3)')
    parser.add_argument(
            '--radius', type=float, default=1.,
            help='Radius of each ball in the environment (default: 1.)')
    parser.add_argument(
            '--gran', type=int, default=2,
            help='Granularity of the simulation in the environment (default: 2)')
    parser.add_argument(
            '--dt', type=float, default=1.,
            help='dt of the simulation computed in each time step. Lower dt leads to a lower percieved env speed (default: 1.)')
    parser.add_argument(
            '--action-force', type=float, default=.3,
            help='action force of the task simulation (default: .3)')

    parser.add_argument(
            '--num-stacked-frames', type=int, default=4,
            help='Number of frames stacked as inputs (default: 4)')
    parser.add_argument(
            '--use-grid', action='store_true', default=False,
            help='whether or not to add grid information to visual input of network')

    parser.add_argument(
            '--use-pretrained-model', action='store_true', default=False,
            help='use the pretrained model as world model'
            )


    args = parser.parse_args()

    assert args.task in ['avoidance', 'maxdist', 'mindist']
    assert not(args.gym and args.use_states) 
    assert not(args.gym and args.use_pretrained_model)
    assert not(args.use_states and args.use_pretrained_model)

    if args.gym:
        pass
    else:
        assert args.env in ['billards']

    return args 

def load_and_encode(path, state_model, device):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    X = torch.tensor(data['X']).float().to(device) 
    actions = torch.tensor(data['action']).float().to(device)
    encoded_state_data = torch.zeros(X.shape[0], X.shape[1] - 8, 3, 18)
    all_appearances = torch.zeros(X.shape[0], X.shape[1] - 8, 3, 3)

    for step in range(len(X[0]) - 8):
        _, init_state, _ = state_model(X[:, step:step+8].permute(0, 1, 4, 3, 2), 0, action=actions[:, step:step+8],
                            pretrain=False, future=None)
        red = torch.argmax(init_state['obj_appearances'][:, -1][:, :, 0], dim=1).view(-1).detach().cpu().numpy() # shape 1000, 3, 3
        others = lambda i: list(set([0, 1, 2]) - set([i]))

        idxs = [[i, *others(i)] for i in red]
        states = init_state['z'][:, -1]
        appearances = init_state['obj_appearances'][:, -1]
        sorted_states = [state[idx] for idx, state in zip(idxs, states)]
        sorted_apps = [state[idx] for idx, state in zip(idxs, appearances)]
        sorted_states = torch.stack(sorted_states)
        sorted_apps = torch.stack(sorted_apps)
        encoded_state_data[:, step] = sorted_states
        all_appearances[:, step] = sorted_apps

    return encoded_state_data.view(-1, *encoded_state_data.shape[2:]), all_appearances.view(-1, *all_appearances.shape[2:])

def save_args(args, path):
    with open(os.path.join(path,'args.json'), 'w') as fp:
        print(f'Saved Args to {os.path.join(path,"args.json")}')
        json.dump(vars(args), fp, sort_keys=True, indent=4)

def load_model():
    user = 'user'
    restore = '/home/{}/Documents/physics/rl/good_rl_run/'.format(user)
    extras = {'nolog':True,
            'traindata': '/home/{}/share/data/billards_w_actions_train_data.pkl'.format(user),
            'testdata': '/home/{}/share/data/billards_w_actions_test_data.pkl'.format(user)}
    trainer = load_vin(extras=extras, restore=restore)
    model = trainer.net
    return model

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gym:
        try:
            env = gym.make(args.env)
            gym_action_size = env.action_space.n
        except:
            print("You did not enter a valid gym control environment; environments must have discrete actions")
            raise NotImplementedError
        task = None

    else:
        greyscale = False
        if args.env == 'billards':
            env = BillardsEnv(n=args.num_balls, r=args.radius, granularity=args.gran, t=args.dt)
        else:
            raise NotImplementedError

        if args.task == 'avoidance':
            task = AvoidanceTask(env, num_stacked=args.num_stacked_frames, action_force=args.action_force, greyscale=greyscale)
        elif args.task == 'maxdist':
            task = MaxDistanceTask(env, num_stacked=args.num_stacked_frames, action_force=args.action_force, greyscale=greyscale)
        elif args.task == 'mindist':
            task = MinDistanceTask(env, num_stacked=args.num_stacked_frames, action_force=args.action_force, greyscale=greyscale)
        else:
            raise NotImplementedError

    model_path = os.path.join(args.model_dir, args.experiment_id)
    summary_path = os.path.join(args.summary_dir, args.experiment_id)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)
    save_args(args, summary_path)
    
    if args.gym:
        actor_critic = Policy(env.observation_space.shape[0], gym_action_size, hidden_size=32, base='control')
    elif args.use_states:
        actor_critic = Policy((env.get_state_shape()[0] * env.get_state_shape()[1]), task.get_action_space(), hidden_size=64, base='control')
    elif args.use_pretrained_model:
        actor_critic = Policy((env.get_state_shape()[0] * env.get_state_shape()[1]), task.get_action_space(), hidden_size=64, base='control')
    else:
        actor_critic = Policy(task.get_framebuffer_shape(), task.get_action_space(), 
                              hidden_size=args.hidden_size, stacked_frames=args.num_stacked_frames, 
                              use_grid=args.use_grid, use_deep_layers=args.use_deep_layers)

    agent = PPOAgent(actor_critic, args.clip_param, args.num_ppo_epochs, 
                     args.num_ppo_mb, args.value_loss_coef, args.entropy_coef,
                     args.lr, args.eps, args.max_grad_norm)

    runner = PPORunner(env=env, task=task, device='cuda', 
                          summary_path=summary_path, 
                          agent=agent, actor_critic=actor_critic,
                          num_steps=args.num_steps, batch_size=args.batch_size, discount=0.95)

    num_batches = int(args.num_env_steps) // args.num_steps // args.batch_size

    try:
        if args.gym:
            run_method = runner.run_gym_batch
        elif args.use_states:
            run_method = runner.run_batch_states  
        elif args.use_pretrained_model:
            run_method = runner.run_pretrained_batch
            state_model = load_model()
        else:
            run_method = runner.run_batch

        user = 'user'
        for i in tqdm(range(num_batches)):
            if run_method.__name__ == 'run_pretrained_batch':
                path = '/home/{}/share/data/billards_w_actions_train_data.pkl'.format(user)
                encoded_state_data, appearences = load_and_encode(path, state_model, 'cuda')
                run_method(state_model, encoded_state_data, appearences)
            else:
                run_method()

            if i % args.save_interval == 0:
                torch.save(actor_critic, os.path.join(model_path, 'model' + str(i) + ".pt"))  

    except:
        torch.save(actor_critic, os.path.join(model_path, 'model' + str(i) + ".pt"))
        print(traceback.format_exc())

    if not os.path.isfile(os.path.join(model_path, 'model' + str(i) + ".pt")):
        torch.save(actor_critic, os.path.join(model_path, 'model' + str(i) + ".pt"))

if __name__ == "__main__":
    args = parse_args()
    main(args)
