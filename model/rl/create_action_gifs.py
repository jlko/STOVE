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


def main():
    r = 1.  # np.array([[1.], [1.], [1.]])
    m = 1.  # np.pi * (r ** 2)
    env = BillardsEnv(granularity=2)
    task = AvoidanceTask(env, action_force=1.)

    # change model file here
    ac = torch.load('./models/avoid_action_f_1/model2000.pt')

    T = 600
    all_imgs = np.zeros((T, *env.get_obs_shape()))

    _init_action = 0

    img, state, reward, done = task.step_frame_buffer(_init_action)
    img = torch.unsqueeze(torch.tensor(img, dtype=torch.float32), dim=0).to('cuda')
    
    for t in tqdm(range(T)):
        value, action, action_log_prob = ac.act(img.permute(0, 3, 1, 2))

        img, state, reward, done = task.step_frame_buffer(action[0].detach().cpu().numpy())

        img = torch.unsqueeze(torch.tensor(img, dtype=torch.float32), dim=0).to('cuda')

        img_helper = img[:, :, :, -3:].cpu().numpy()

        if reward < 0:
            img_helper = 1 - img_helper
        all_imgs[t] = img_helper

    all_imgs = (255 * all_imgs).astype(np.uint8)
    imageio.mimsave('./output.gif', all_imgs, fps=24)

if __name__ == "__main__":
    main()
