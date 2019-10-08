import numpy as np 
import imageio
from tqdm import tqdm
import torch 

# TODO: this is ugly as hell but python sucks sometimes, should try to put everything in packages?
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from envs import BillardsEnv, AvoidanceTask
from mcts_env import MCTS


def main():
    ac = torch.load('./models/no_deep/model2400.pt')
    num_runs = 50

    T = 100

    ppo_reward = []
    mcts_reward = []

    for run in tqdm(range(num_runs)):
        r = 1.  # np.array([[1.], [1.], [1.]])
        m = 1.  # np.pi * (r ** 2)
        hw = 10
        env = BillardsEnv(granularity=2, n=3, hw=10, seed=run)
        task = AvoidanceTask(env, action_force=0.6)

        env_mcts = BillardsEnv(granularity=2, n=3, hw=10, seed=run)
        task_mcts = AvoidanceTask(env_mcts, action_force=0.6)

        # change model file here
        
        _init_action = 0

        img, state, reward, done = task.step_frame_buffer(_init_action)
        img = torch.unsqueeze(torch.tensor(img, dtype=torch.float32), dim=0).to('cuda')

        task_mcts.step(_init_action)
        

        for t in tqdm(range(T)):
            mcts = MCTS(task_mcts, max_rollout=30)
            value, action, action_log_prob = ac.act(img.permute(0, 3, 1, 2))

            img, state, reward, done = task.step_frame_buffer(action[0].detach().cpu().numpy())
            img = torch.unsqueeze(torch.tensor(img, dtype=torch.float32), dim=0).to('cuda')

            ppo_reward.append(reward)

            action = mcts.run_mcts(300)
            _, _, reward, _ = task_mcts.step(action)

            mcts_reward.append(reward)

        print(f'MCTS running: {np.mean(mcts_reward)}')
        print(f'PPO running: {np.mean(ppo_reward)}')
    
    print('MCTS')
    print(np.mean(mcts_reward))
    print(np.std(mcts_reward))

    print('PPO')
    print(np.mean(ppo_reward))
    print(np.std(ppo_reward))


if __name__ == "__main__":
    main()
