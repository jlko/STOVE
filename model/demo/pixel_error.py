import torch
import itertools
import numpy as np
import imageio

import argparse
import os
import glob

from vin import main
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def run_fmt(x, with_under=False):
    return 'run{:03d}'.format(x) if not with_under else 'run_{:03d}'.format(x)

def get_pixel_error(restore, linear=False, path=''):

    extras = {'nolog':True}

    self = main(restore=restore, extras=extras)

    if self.c.supairvised is True:
        return None

    # make sure they all access the same data!
    print(self.c.testdata)

    step = self.c.frame_step
    visible = self.c.num_visible
    batch_size = self.c.batch_size
    skip = self.c.skip

    # make sure this is the same
    print(step, visible, batch_size, skip)

    long_rollout_length = self.c.num_frames // step - visible

    total_images = self.test_dataset.total_img
    total_labels = self.test_dataset.total_data

    # apply step and batch size once
    total_images = total_images[:batch_size, ::step]
    total_labels = total_labels[:batch_size, ::step]

    # true data to compare against
    true_images = total_images[:, skip:(visible+long_rollout_length)]
    true_images = torch.tensor(true_images).to(self.c.device).type(self.c.dtype)

    # First obtain reconstruction of input.
    vin_input = total_images[:, :visible]
    vin_input = torch.tensor(vin_input).to(self.c.device).type(self.c.dtype)
    elbo, prop_dict2 = self.net(vin_input, self.c.plot_every)

    z_recon = prop_dict2['z']
    # Use last state to do rollout
    if not linear:
        z_pred = self.net.rollout(z_recon[:, -1], long_rollout_length)
    else:
        # propagate last speed
        v = z_recon[:, -1, :, 4:6].unsqueeze(1).repeat(1, long_rollout_length, 1, 1)
        t = torch.arange(1, long_rollout_length+1).repeat(
                v.shape[0], *v.shape[2:], 1).permute(0,3,1,2).double()
        dx = v * t
        new_x = z_recon[:, -1, :, 2:4].unsqueeze(1).repeat(1, long_rollout_length, 1, 1) \
                + dx
        z_pred = torch.cat(
                [z_recon[:, -1, :, :2].unsqueeze(1).repeat(1, long_rollout_length, 1, 1),
                 new_x,
                 v,
                 z_recon[:, -1, :, 6:].unsqueeze(1).repeat(1, long_rollout_length, 1, 1)],
                 -1
                )
    # concat
    z_seq = torch.cat([z_recon, z_pred], 1)
    # sigmoid positions to -0.8, 0.8 to make errors comparable
    if linear:
        print('clamp positions to 0.9')
        z_seq = torch.cat([
            z_seq[..., :2],
            torch.clamp(z_seq[..., 2:4], -0.9, 0.9),
            z_seq[..., 6:]], -1)

    # use mpe to get reconstructed images
    model_images = self.net.reconstruct_from_z(z_seq)
    model_images = torch.clamp(model_images, 0, 1)
    plot_sample = model_images[:10, :, 0].detach().cpu().numpy()
    print(plot_sample.max(), plot_sample.min())
    plot_sample = (255 * plot_sample.reshape(-1, self.c.height, self.c.width)).astype(np.uint8)
    
    filename = 'linear_' if linear else '' 
    filename += 'pixel_error_sample.gif'
    filepath = os.path.join(path, filename) 
    print('Saving gif to ', filepath)
    imageio.mimsave(
        filepath, plot_sample, fps=24)

    mse = torch.mean(((true_images - model_images)**2), dim=(0, 2, 3, 4))
    
    # also log state differences
    # bug_potential... for some reason self.c.coord_lim is 30 but max true_states is 10 
    # for gravity
    true_states = total_labels[:, skip:(visible+long_rollout_length)]/10 * 2 -1
    true_states = torch.tensor(true_states).to(self.c.device).type(self.c.dtype)
    permutations = list(itertools.permutations(range(0, self.c.num_obj)))
    errors = []
    for perm in permutations:
        errors += [torch.sqrt(
            ((true_states[:, :5, :, :2]-z_seq[:, :5, perm, 2:4])**2).sum(-1)).mean((1, 2))]
    errors = torch.stack(errors, 1)
    _, idx = errors.min(1)
    selector = list(zip(range(idx.shape[0]), idx.cpu().tolist()))
    pos_matched = [z_seq[i, :, permutations[j]] for i, j in selector]
    pos_matched = torch.stack(pos_matched, 0)

    mse_states = torch.sqrt(((
        true_states[..., :2] - pos_matched[..., 2:4])**2).sum(-1)).mean((0, 2))
    return mse, mse_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str,
            help="set folder from which to create pixel errors for, must contain runs of model")
    parser.add_argument('--linear', action='store_true',
            help='create linear errors')
    parser.add_argument('--no_save', action='store_true')
    args = parser.parse_args()
    ## define params
    filename = 'pixel_errors.csv'
    if args.linear:
        filename = 'linear_' + filename
    
    if not 'run' in args.path[-10:]:
        restores = glob.glob(args.path+'run*')
        restores = sorted(restores)
    else:
        restores = [args.path]

    print(restores)
    for restore in restores:
        try:
            mse, mse_states = get_pixel_error(restore, args.linear, args.path)
        except Exception as e:
            print(e)
            print('Not possible for run {}.'.format(restore))
            continue

        mse =  mse.cpu().detach().numpy()
        if args.no_save:
            continue
        with open(os.path.join(args.path, 'test',  filename), 'a') as f:
            f.write(','.join(['{:.6f}'.format(i) for i in mse])+'\n')
        with open(os.path.join(args.path, 'test',  'states_'+filename), 'a') as f:
            f.write(','.join(['{:.6f}'.format(i) for i in mse_states])+'\n')

