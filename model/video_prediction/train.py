import os
import itertools
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch import autograd
from torch.utils.data import DataLoader
from torch.distributions import Uniform

from ..visualize import plot_positions, animate, plot_boxes, plot_grad_flow, plot_patches, plot_bg
from ..utils.utils import ExperimentLogger
from ..utils import utils

class Trainer:
    def __init__(self, config, net, train_dataset, test_dataset):
        self.net = net
        self.params = net.parameters()
        self.c = config

        self.dataloader = DataLoader(train_dataset,
                                     batch_size=self.c.batch_size,
                                     shuffle=True,
                                     num_workers=self.c.num_workers,
                                     drop_last=True)

        self.logger = ExperimentLogger(self.c)

        if test_dataset is not None:
            self.test_dataset = test_dataset
            self.test_dataloader = DataLoader(test_dataset,
                                              batch_size=self.c.batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              drop_last=True)

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.c.learning_rate,
            amsgrad=self.c.debug_amsgrad)

        if self.c.load_supair is not None:
            self.load_supair()

        if not self.c.supair_grad:
            # may not even be necessary
            self.disable_supair_grad()

        self.epoch_start, self.step_start = None, None
        if self.c.checkpoint_path is not None:
            self.load()

        self.z_types = ['z', 'z_sup', 'z_vin'] if not self.c.supair_only else ['z']

        if self.c.action_conditioned:
            if not self.c.debug_mse:
                self.reward_loss = nn.BCELoss()
            else:
                self.reward_loss = nn.MSELoss()
        else:
            self.reward_loss = lambda x, y: 0

    def save(self, epoch, step):
        path = os.path.join(self.logger.exp_dir, "checkpoint")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
        print('Parameters saved to {}'.format(self.logger.exp_dir))

    def load(self):

        checkpoint = torch.load(
            self.c.checkpoint_path, map_location=self.c.device)

        # stay compatible with old loading
        if 'model_state_dict' in checkpoint.keys():
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            step = checkpoint['step']
            self.epoch_start = epoch
            self.step_start = step

        else:
            self.net.load_state_dict(checkpoint)

        print('Parameters loaded from {}.'.format(self.c.checkpoint_path))


    def load_supair(self):
        """ Load weights of supair. Right now this does not! load all weights of checkpoint.

        Adapted from discuss.pytorch.org/t/23962.
        """
        pretrained_dict = torch.load(self.c.load_supair, map_location=self.c.device)

        if 'model_state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['model_state_dict']            

        model_dict = self.net.state_dict()

        # 1. filter out unnecessary keys
        # only load spn
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and any(i in k for i in ['spn', 'encoder'])}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.net.load_state_dict(model_dict)

        print('Successfully loaded the following supair parameters from checkpoint {}:'.format(
            self.c.load_supair))
        print(pretrained_dict.keys())

    def disable_supair_grad(self):
        for p in self.net.obj_spn.parameters():
            p.requires_grad = False
        for p in self.net.bg_spn.parameters():
            p.requires_grad = False
        for p in self.net.encoder.parameters():
            p.requires_grad = False
        for n, p in self.net.named_parameters():
            print('gradients for ', n, p.requires_grad)\

    def prediction_error(
        self, predicted, true,
        return_velocity=True, return_id_swaps=True, return_full=False,
        return_matched=False, level='sequence'):
        """Loss of predicted position by encoder.

        :param predicted: shape (n, T, o, 4)
        :param true: shape (n, T, o, 4). Positions are on [0:2], velocities on
        [2:4]. Can check since current_position + current_velocity = next_position.
        :param level: Level at which to match the true and predicted object ordering
            for Physics prediciton, we should only allow one global permutation of
            the predicted object ordering over the sequence, since we want to be
            aware of id swaps. For SuPair training, we may want to get the error


        Coordinate system of true is from [0, 10], for predicted in [-1, 1].
        Scale true accordingly.
        """
        if self.c.supair_only:
            return_velocity = False
            level = 'image'
        # use at moste the first 4 time values for assigning object ordering
        T = min(4, predicted.shape[1])

        # ignore velocities (if available)
        pos_pred = predicted[..., :2]
        # Transform true positions to scale of predicted.
        # from visual inspection this seems to work best (ignore velocities)
        pos_true = 2 * (true[..., :2] / 10.0) - 1.0

        errors = []
        permutations = list(itertools.permutations(range(0, self.c.num_obj)))

        if level == 'sequence':
            # unsupervised learning cannot be punished if it gets object order wrong
            # therefore get all n! object combinations and take lowest error for
            # a given sequence!

            for perm in permutations:
                errors += [torch.sqrt(((pos_pred[:, :T, perm] - pos_true[:, :T])**2).sum(-1)).mean((1, 2))]
                # sum_k/T(sum_j/o(root(sum_i((x_i0-x_i1)**2))))
                # sum_i over x and y coordinates -> root(sum squared) is distance of
                # objects for that permutation. sum j is then over all objects in image
                # and sum_k over all images in sequence. that way we do 1 assignment
                # of objects over whole sequence! this loss will now punish id swaps
                # over sequence. sum_j and _k are mean. st. we get the mean distance
                # to true position

            # shape (n, o!)
            errors = torch.stack(errors, 1)

            # sum to get error per image
            _, idx = errors.min(1)
            # idx now contains a single winning permutation per sequence!

            selector = list(zip(range(idx.shape[0]), idx.cpu().tolist()))

            pos_matched = [pos_pred[i, :, permutations[j]] for i, j in selector]
            pos_matched = torch.stack(pos_matched, 0)

        elif level == 'image':
            pos_pred_f = pos_pred.flatten(end_dim=1)
            pos_true_f = pos_true.flatten(end_dim=1)
            for perm in permutations:
                errors += [torch.sqrt(((pos_pred_f[:, perm] - pos_true_f)**2).sum(-1)).mean((1))]
            errors = torch.stack(errors, 1)
            _, idx = errors.min(1)
            selector = list(zip(range(idx.shape[0]), idx.cpu().tolist()))
            pos_matched = [pos_pred_f[i, permutations[j]] for i, j in selector]
            pos_matched = torch.stack(pos_matched, 0)
            pos_matched = pos_matched.reshape(predicted[..., :2].shape)
        else:
            raise ValueError

        res = {}
        if not return_full:
            min_errors = torch.sqrt(((pos_matched - pos_true)**2).sum(-1)).mean((1, 2))
            res['error'] = min_errors.mean().cpu()
            res['std_error'] = min_errors.std().cpu()

        else:
            # return error over sequence!
            # get correctly matched sequence
            error_over_sequence = torch.sqrt(((pos_matched - pos_true)**2).sum(-1)).mean(-1)
            res['error'] = error_over_sequence.mean(0).cpu()
            res['std_error'] = error_over_sequence.std(0).cpu()

        if return_velocity:

            # get velocities and transform
            vel_pred = predicted[..., 2:4]
            vel_true = true[..., 2:4] / 10.0 * 2.0

            # get correctly matched velocities
            vel_matched = [vel_pred[i, :, permutations[j]] for i, j in selector]
            vel_matched = torch.stack(vel_matched, 0)

            # again. root of sum of squared distances per object. mean over image.
            v_errors = torch.sqrt(((vel_true - vel_matched)**2).sum(-1)).mean(-1)

            if not return_full:
                res['v_error'] = v_errors.mean().cpu()
                res['std_v_error'] = v_errors.std().cpu()

            else:
                # do mean along all images in sequence
                res['v_error'] = v_errors.mean(0).cpu()
                res['std_v_error'] = v_errors.std(0).cpu()

        if return_matched:
            res['pos_matched'] = pos_matched
            res['vel_matched'] = vel_matched

        if return_id_swaps:
            # Do min over img instead of over sequence. This is equiv to old
            #  way of calculating error.
            errors = []
            for perm in permutations:
                # this is distance of object pairing
                predicted_f = pos_pred.flatten(end_dim=1)
                true_f = pos_true.flatten(end_dim=1)
                errors += [torch.sqrt(((predicted_f[:, perm] - true_f)**2).sum(-1))]
            errors = torch.stack(errors, 1)
            # mean per img then min along stack axis
            _, idx = errors.mean(-1).min(1)

            # how many sequences contain more than one object ordering
            idx = idx.reshape(true.shape[:2])[:, :]
            id_swaps = torch.chunk(idx, idx.shape[1], 1)
            id_swaps = [i.squeeze() for i in id_swaps]
            # compare each t to t+1 in terms of minimum indices, returns n times 0 or 1
            # for each comparison
            id_swaps = [id_swaps[i] == id_swaps[i+1] for i in range(len(id_swaps)-1)]
            id_swaps = torch.stack(id_swaps, 1)
            # if each is the same as its neighbor, the whole seq is the same (1)
            # if not, there are zeros in prod
            id_swaps = torch.prod(id_swaps, 1)
            # sum to get number instead of 0, 1 list
            id_swaps = id_swaps.sum()

            id_swap_percentage = (idx.shape[0] - id_swaps.cpu().double()) / idx.shape[0]
            res['swaps'] = id_swap_percentage

        return res

    def plot_results(self, step_counter, images, prop_dict, future=False):
        # just give first  sequences
        n_seq = self.c.n_plot_sequences
        plot_images = images[:n_seq].detach().cpu().numpy()
        # ignore velocities if available
        plot_z = prop_dict['z'].flatten(end_dim=1)[..., :4].cpu().numpy()

        if not self.c.nolog:
            add = '_rollout' if future else ''
            save_path = os.path.join(self.logger.img_dir, '{:05d}{}.png'.format(step_counter, add))
        else:
            save_path = None

        plot_boxes(
            plot_images, plot_z,
            self.c.width,
            self.c.height,
            n_sequences=n_seq,
            future=future,
            save_path=save_path,
            visdom=self.c.visdom
            )

        if self.c.debug_extend_plots:
            overlap = prop_dict['overlap_ratios'].cpu().numpy()
            marg_patch = prop_dict['marginalise_flat'].detach().reshape(
                (-1, self.c.patch_width, self.c.patch_height)).cpu().numpy()

            marginalise_bg = prop_dict['marginalise_bg'].cpu().numpy()
            bg_loglik = prop_dict['bg_loglik'].cpu().numpy()

            plot_bg(marginalise_bg, bg_loglik, n_sequences=n_seq, visdom=self.c.visdom)

            patches = prop_dict['patches'].cpu().numpy()
            patches_ll = prop_dict['patches_loglik'].cpu().numpy()

            plot_patches(patches, marg_patch, overlap, patches_ll, self.c, visdom=self.c.visdom)

    def adjust_learning_rate(self, optimizer, value, epoch, step):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        #lr = self.c.learning_rate * (0.7 ** (epoch // 50))
        lr = self.c.learning_rate * np.exp(-step / value)
        lr = max(lr, 0.0002)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        print('Starting training for {}'.format(self.c.description))
        print('Only pretraining.' if self.c.supair_only else 'Full inference.')
        # create some more rollouts at the end of training
        skip = self.c.skip

        start_epoch = self.epoch_start if self.epoch_start is not None else 0
        step_counter = self.step_start if self.step_start is not None else 0

        if self.c.debug_bw:
            assert self.c.channels == 1, ValueError('Warning: need to set channels to 1 in config be4')
        start = time.time()
        self.test(step_counter, start)

        for epoch in range(start_epoch, self.c.num_epochs):
            for i, data in enumerate(self.dataloader, 0):

                if self.c.debug_anneal_lr:
                    self.adjust_learning_rate(self.optimizer, self.c.debug_anneal_lr, epoch, step_counter)

                now = time.time() - start
                step_counter += 1
                images = data['present_images'].to(self.c.device).type(self.c.dtype)
                if self.c.action_conditioned:
                    actions = data['present_actions'].to(self.c.device).type(self.c.dtype)
                else:
                    actions = None

                self.optimizer.zero_grad()

                elbo, prop_dict, rewards = self.net(
                    images,
                    step_counter,
                    actions,
                    self.c.supair_only)
                min_ll = -1.0 * elbo

                if self.c.action_conditioned:
                    target_rewards = data['present_rewards'][:, self.c.skip:].to(self.c.device).type(self.c.dtype)
                    mse_rewards = self.reward_loss(rewards.flatten(), target_rewards.flatten())

                    if self.c.debug_reward_rampup is not False:
                        reward_weight = min(1, step_counter/self.c.debug_reward_rampup)
                    else:
                        reward_weight = 1

                    reward_factor = self.c.debug_reward_factor
                    min_ll = min_ll + reward_factor * reward_weight * mse_rewards
                else:
                    mse_rewards = torch.Tensor([0])

                min_ll.backward()

                if self.c.debug_gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optimizer.step()

                # Plot examples
                if step_counter % self.c.plot_every == 0:
                    plot_grad_flow(self.net.named_parameters())
                    # full states only available after some steps
                    plot_images = images[:, self.c.skip:]

                    self.plot_results(step_counter, plot_images, prop_dict)

                # Print and log performance
                if step_counter % self.c.print_every == 0:
                    self.error_and_log(
                        elbo.item(), mse_rewards.item(), min_ll.item(), prop_dict, data, step_counter, now)

                # Save parameters
                if step_counter % self.c.save_every == 0:
                    self.save(epoch, step_counter)

                if self.c.debug_code and not self.c.supair_only:
                    self.test(step_counter, now)

                if self.c.debug_test_mode:
                    break

            # Test performance each epoch
            if not self.c.supair_only:
                self.test(step_counter, start)

            print("Epoch: ", epoch, " finished.")

            if self.c.debug_test_mode:
                break

        # create some more rollouts at the end of training
        if not self.c.debug_test_mode:
            for idx in self.c.rollout_idxs:
                self.long_rollout(step_counter, idx)

        if not self.c.nolog:
            self.save(epoch, step_counter)
            succ = os.path.join(self.logger.exp_dir, 'success')
            open(succ, 'w').close()
        print('Finished Training!')

    def error_and_log(self, elbo, reward, min_ll, prop_dict, data, step_counter, now, add=''):
        # Present
        skip = self.c.skip

        # align true data with timesteps for which we have predictions
        perf_dict = {'step': step_counter, 'time': now, 'elbo': elbo, 'reward': reward, 'min_ll': min_ll}

        # non z entries
        other_keys = list(filter(lambda x: x[0] != 'z', list(prop_dict.keys())))
        other_dict = {key: prop_dict[key] for key in other_keys}
        perf_dict.update(other_dict)

        z_true = data['present_labels'][:, skip:]
        z_true = z_true.to(self.c.device).type(self.c.dtype)
        for z in self.z_types:
            if z in ['z', 'z_sup']:
                # have scales and need to ignore for prediction_error
                predicted = prop_dict[z][..., 2:]
                scales = prop_dict[z].flatten(end_dim=2)[:, :2].mean(0)
                perf_dict['scale_x'] = scales[0]
                perf_dict['scale_y'] = scales[1]
            else:
                predicted = prop_dict[z]
                perf_dict['scale_x'] = float('nan')
                perf_dict['scale_y'] = float('nan')

            error_dict = self.prediction_error(
                predicted, z_true)
            perf_dict.update(error_dict)

            z_std = prop_dict[z+'_std']
            for i, std in enumerate(z_std):
                perf_dict[z+'_std_{}'.format(i)] = std
            perf_dict['type'] = z + add

            self.logger.performance(
                perf_dict)

    @torch.no_grad()
    def test(self, step_counter, start):
        """Evaluate performance on test_data rollout.

        Also always create one long rollout gif.

        """
        self.net.eval()
        now = time.time() - start
        for i, data in enumerate(self.test_dataloader, 0):

            present = data['present_images']
            present = present.to(self.c.device).type(self.c.dtype)
            if self.c.action_conditioned:
                actions = data['present_actions'].to(self.c.device).type(self.c.dtype)
            else:
                actions = None

            elbo, prop_dict, rewards = self.net(
                present,
                self.c.plot_every,
                actions,
                self.c.supair_only
                )

            min_ll = -1.0 * elbo

            if self.c.action_conditioned:
                target_rewards = data['present_rewards'][:, self.c.skip:].to(self.c.device).type(self.c.dtype)
                mse_rewards = self.reward_loss(rewards.flatten(), target_rewards.flatten())

                if self.c.debug_reward_rampup is not False:
                    reward_weight = min(1, step_counter/self.c.debug_reward_rampup)
                else:
                    reward_weight = 1

                reward_factor = self.c.debug_reward_factor
                min_ll = min_ll + reward_factor * reward_weight * mse_rewards
            else:
                mse_rewards = torch.Tensor([0])

            self.error_and_log(
                elbo.item(), mse_rewards.item(), min_ll.item(), prop_dict, data, step_counter, now, add='_roll')

            if self.c.action_conditioned:
                future_actions = data['future_actions'].to(self.c.device).type(self.c.dtype)
                future_rewards = data['future_rewards'].to(self.c.device).type(self.c.dtype)
            else:
                future_actions = None
                future_rewards = None

            if self.c.debug_core_appearance:
                appearances = prop_dict['obj_appearances'][:, -1]
            else:
                appearances = None

            z_pred, rewards_pred = self.net.rollout(
                prop_dict['z'][:, -1], actions=future_actions,
                appearance=appearances)

            if self.c.action_conditioned:
                # add code for rewards on rollout
                # future rollout_error
                future_reward_loss = self.reward_loss(rewards_pred.flatten(), future_rewards.flatten())
            else:
                future_reward_loss = 0

            z_true = data['future_labels'].to(self.c.device).type(self.c.dtype)
            error_dict = self.prediction_error(
                z_pred[..., 2:], z_true)

            perf_dict = {'step': step_counter, 'time': now, 'elbo': elbo, 'reward': future_reward_loss}
            perf_dict.update(error_dict)
            other_keys = list(filter(lambda x: x[0] != 'z', list(prop_dict.keys())))
            other_dict = {key: prop_dict[key] for key in other_keys}
            perf_dict.update(other_dict)
            perf_dict['type'] = 'rollout'

            self.logger.performance(perf_dict)

            if self.c.debug_code:
                break
            if i > 7:
                break

        self.long_rollout(step_counter)
        self.net.train()

    @torch.no_grad()
    def long_rollout(self, step_counter, idx=0, with_logging=True, actions=None):
        """Create one long rollout and save it as an animated GIF.

        :param idx: Index of sequence in test data set.
        """
        self.net.eval()

        step = self.c.frame_step
        visible = self.c.num_visible
        batch_size = self.c.batch_size
        skip = self.c.skip
        long_rollout_length = self.c.num_frames // step - visible

        total_images = self.test_dataset.total_img
        total_labels = self.test_dataset.total_data

        if self.c.action_conditioned:
            if actions is None:
                total_actions = self.test_dataset.total_actions
            else:
                total_actions = actions

            total_actions = total_actions[:batch_size, ::step]
            action_input = total_actions[:, :visible]
            action_input = torch.tensor(action_input).to(self.c.device).type(self.c.dtype)

            total_rewards = self.test_dataset.total_rewards
            total_rewards = total_rewards[:batch_size, ::step]
            real_rewards = total_rewards[idx, self.c.skip:, 0]

            # need some actions for rollout
            true_future_actions = total_actions[:, visible:(visible+long_rollout_length)]
            true_future_actions = torch.tensor(true_future_actions).to(self.c.device).type(self.c.dtype)

            action_recon = total_actions[idx:idx+2, :(visible+long_rollout_length)]
            action_recon = torch.tensor(action_recon, device=self.c.device, dtype=self.c.dtype)

        else:
            action_input = None
            true_future_actions = None
            action_recon = None

        # apply step and batch size once
        total_images = total_images[:batch_size, ::step]
        total_labels = total_labels[:batch_size, ::step]


        # First obtain reconstruction of input.
        vin_input = total_images[:, :visible]
        vin_input = torch.tensor(vin_input).to(self.c.device).type(self.c.dtype)

        elbo, prop_dict2, rewards_recon = self.net(vin_input, self.c.plot_every, action_input)

        z_recon = prop_dict2['z']

        # Use last state to do rollout
        if self.c.debug_core_appearance:
            appearances = prop_dict2['obj_appearances'][:, -1]
        else:
            appearances = None

        z_pred, rewards_pred = self.net.rollout(
            z_recon[:, -1], num=long_rollout_length, actions=true_future_actions,
            appearance=appearances)

        # Plot first 8 imgs for each in batch.
        true_future = total_images[:, visible:(visible+long_rollout_length)]
        true_future = torch.tensor(true_future).to(self.c.device).type(self.c.dtype)
        prop_dict2['z'] = z_pred[:, :8].detach()
        # self.plot_results(step_counter, true_future[:, :8], prop_dict2, future=True)

        # Plot complete sequence!
        simu_recon = z_recon.detach()
        simu_rollout = z_pred.detach()
        simu = torch.cat([simu_recon, simu_rollout], 1)

        if not self.c.nolog and with_logging:
            # Get prediction error over long sequence for loogging
            real_labels = torch.Tensor(total_labels, device=self.c.device).type(self.c.dtype)
            real_labels = real_labels[:, skip:(visible+long_rollout_length)]
            error_dict = self.prediction_error(
                simu[..., skip:], real_labels, return_velocity=True, return_full=True,
                return_id_swaps=False)

            for name, data in error_dict.items():
                file = os.path.join(self.logger.exp_dir, '{}.csv'.format(name))
                with open(file, 'a') as f:
                    f.write(','.join(['{:.6f}'.format(i) for i in data])+'\n')

        # only select positions and translate to [0, 10] frame
        simu = (simu[idx, :, :, 2:4] + 1.0) / 2.0 * self.c.coord_lim
        simu = simu.detach().cpu().numpy()

        # also get a reconstruction of z along entire sequence
        vin_input = total_images[idx:idx+2, :(visible+long_rollout_length)]
        vin_input = torch.tensor(vin_input, device=self.c.device, dtype=self.c.dtype)

        elbo, prop_dict3, recon_reward = self.net(vin_input, self.c.plot_every, actions=action_recon)

        recon = (prop_dict3['z'][0, :, :, 2:4] + 1.0) / 2.0 * self.c.coord_lim
        recon = recon.cpu().numpy()
        recon_reward = recon_reward.cpu().numpy()[0].squeeze()

        # real = (total_labels[idx, self.c.skip:, :, :2] + 1.0) / 2.0 * 10.0
        real = total_labels[idx, self.c.skip:, :, :2] / 10 * self.c.coord_lim

        if self.c.action_conditioned:
            # add rewards to gif
            rewards_model = torch.cat([rewards_recon, rewards_pred], 1).squeeze()[idx]
            rewards_model = rewards_model.detach().cpu().numpy()

        # Make Gifs.
        # we are loosing the first 2 images
        if not self.c.nolog:
            gif_path = os.path.join(self.logger.exp_dir, 'gifs')
        else:
            gif_path = os.path.join(self.c.experiment_dir, 'tmp')

        print("Make GIFs in {}".format(gif_path))

        gifs = [real, simu, simu, recon, recon]
        if self.c.action_conditioned:
            rewards = [real_rewards, rewards_model, rewards_model, recon_reward, recon_reward]
        else:
            rewards = len(gifs) * [None]

        names = ['real_{:02d}'.format(idx),
                 'rollout_{:02d}'.format(idx),
                 'rollout_{:02d}_{:05d}'.format(idx, step_counter),
                 'recon_{:02d}'.format(idx),
                 'recon_{:02d}_{:05d}'.format(idx, step_counter)]

        for gif, name, reward in zip(gifs, names, rewards):
            animate(
                gif, gif_path, name, size=self.c.coord_lim,
                res=self.c.width, r=self.c.r, rewards=reward)



        print("Done")
        # Set net to train mode again.
        self.net.train()
