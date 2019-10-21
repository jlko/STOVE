"""Contains class for model training."""

import os
import itertools
import time
import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from ..utils.visualize import (
    animate, plot_boxes, plot_grad_flow, plot_patches, plot_bg)
from ..utils.utils import ExperimentLogger


class AbstractTrainer:
    """Abstract trainer class.

    Exists, s.t. Trainer can share code between STOVE and supervised approach.
    """
    def __init__(self, config, stove, train_dataset, test_dataset):
        """Set up abstract trainer."""
        self.stove = stove
        self.params = stove.parameters()

        if config.debug_test_mode:
            config.print_every = 1
            config.plot_every = 1

        self.c = config

        # implemented as property, s.t. train_dataset can easily be overwritten
        # from the outside for mcts loop training
        self.dataloader = train_dataset
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.c.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)

        self.optimizer = optim.Adam(
            self.stove.parameters(),
            lr=self.c.learning_rate,
            amsgrad=self.c.debug_amsgrad)

        if self.c.load_encoder is not None:
            self.load_encoder()

        if not self.c.supair_grad:
            self.disable_encoder_grad()

        # if we restore from checkpoint, also restore epoch and step
        self.epoch_start, self.step_start = 0, 0
        if self.c.checkpoint_path is not None:
            self.load()

    @property
    def dataloader(self):
        """Return train_dataset if set already."""
        return self._train_dataset

    @dataloader.setter
    def dataloader(self, train_dataset):
        """Set train_dataset by wrapping DataLoader."""
        self._train_dataset = DataLoader(
            train_dataset,
            batch_size=self.c.batch_size,
            shuffle=True,
            num_workers=self.c.num_workers,
            drop_last=True)

    def save(self, epoch, step):
        """Save model dict, optimizer and progress indicator."""
        path = os.path.join(self.logger.exp_dir, "checkpoint")

        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.stove.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path + '_{}'.format(step))


        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.stove.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)

        print('Parameters saved to {}'.format(self.logger.exp_dir))

    def load(self):
        """Load model dict from checkpoint."""
        checkpoint = torch.load(
            self.c.checkpoint_path, map_location=self.c.device)

        # stay compatible with old loading
        if 'model_state_dict' in checkpoint.keys():
            self.stove.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            step = checkpoint['step']
            self.epoch_start = epoch
            self.step_start = step

        else:
            self.stove.load_state_dict(checkpoint)

        print('Parameters loaded from {}.'.format(self.c.checkpoint_path))

    def load_encoder(self):
        """Load weights of encoder.

        Adapted from discuss.pytorch.org/t/23962.
        """
        pretrained_dict = torch.load(
            self.c.load_encoder, map_location=self.c.device)
        pretrained_dict = pretrained_dict['model_state_dict']

        model_dict = self.stove.state_dict()
        # 1. filter out unnecessary keys, only load spn
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and 'encoder' in k}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.stove.load_state_dict(model_dict)

        print('Loaded the following supair parameters from {}:'.format(
            self.c.load_encoder))
        print(pretrained_dict.keys())

    def disable_supair_grad(self):
        """Disable gradient for SuPAIR if desired."""
        for p in self.stove.obj_spn.parameters():
            p.requires_grad = False
        for p in self.stove.bg_spn.parameters():
            p.requires_grad = False
        for p in self.stove.encoder.parameters():
            p.requires_grad = False
        for n, p in self.stove.named_parameters():
            print('Gradients for {} enabled: {}'.format(n, p.requires_grad))

    def prediction_error(self):
        """Abstract method."""
        raise ValueError('Needs to be overwritten by derived class.')

    def plot_results(self):
        """Abstract method."""
        raise ValueError('Needs to be overwritten by derived class.')

    def test(self):
        """Abstract method."""
        raise ValueError('Needs to be overwritten by derived class.')

    def train(self):
        """Abstract method."""
        raise ValueError('Needs to be overwritten by derived class.')

    def long_rollout(self):
        """Abstract method."""
        raise ValueError('Needs to be overwritten by derived class.')


class Trainer(AbstractTrainer):
    """Trainer for model optimization.

    Fully compatible with STOVE as well as action-conditioned STOVE.
    """

    def __init__(self, config, stove, train_dataset, test_dataset):
        """Set up trainer.

        This is conveniently called with main.py. Given a valid config, main.py
        takes care of initalising the trainer, model and dataset.

        Do not modify the config object.
        """
        super().__init__(config, stove, train_dataset, test_dataset)

        self.logger = ExperimentLogger(self.c)

        # differentiate between z from dynamics , z from supair, and
        # combined z
        self.z_types = ['z', 'z_sup', 'z_dyn'] if not self.c.supair_only else ['z']

        if self.c.action_conditioned:
            if not self.c.debug_mse:
                self.reward_loss = nn.BCELoss()
            else:
                self.reward_loss = nn.MSELoss()


    def prediction_error(self, predicted, true,
                         return_velocity=True, return_id_swaps=True,
                         return_full=False, return_matched=False,
                         level='sequence'):
        """Error of predicted positions and velocities against ground truth.

        Args:
            predicted (torch.Tensor), (n, T, o, 4): Stoves positions and
                velocities.
            true (torch.Tensor), (n, T, o, 4): Positions and velocities from env.
            return_velocity (bool): Return velocity errors.
            return_id_swaps (bool): Return percentage of id swaps over sequence.
            return_full (bool): Return errors over T dimension.
            return_matched (bool): Return permuted positions and velocities.
            level (str): Set to 'sequence' or 'imgage'. Object orderings for a
                sequence are not aligned, b/c model is unsupervised. Need to be
                matched. Specify level at which to match the true and predicted
                object ordering. For physics prediciton, we should only allow
                one global permutation of the predicted object ordering over the
                sequence, since we want id swaps to affect the error. For SuPair
                (only) training, we want to get the error per image.

        Returns:
            res (dict): Results dictionary containing errors, as set by the
                above flags.

        """
        if self.c.supair_only:
            return_velocity = False
            level = 'image'

        # use at moste the first 4 time values for assigning object ordering
        T = min(4, predicted.shape[1])

        pos_pred = predicted[..., :2]
        pos_true = true[..., :2]

        errors = []
        permutations = list(itertools.permutations(range(0, self.c.num_obj)))

        if level == 'sequence':
            # unsupervised learning cannot be punished if it gets object order wrong
            # therefore get all n! object combinations and take lowest error for
            # a given sequence!

            for perm in permutations:
                error = ((pos_pred[:, :T, perm] - pos_true[:, :T])**2).sum(-1)
                error = torch.sqrt(error).mean((1, 2))
                errors += [error]
                """sum_k/T(sum_j/o(root(sum_i((x_i0-x_i1)**2))))
                   sum_i over x and y coordinates -> root(sum squared) is
                   distance of objects for that permutation. sum j is then over
                   all objects in image and sum_k over all images in sequence.
                   that way we do 1 assignment of objects over whole sequence!
                   this loss will now punish id swaps over sequence. sum_j and
                   _k are mean. st. we get the mean distance to true position
                """

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
                errors += [torch.sqrt(
                    ((pos_pred_f[:, perm] - pos_true_f)**2).sum(-1)).mean((1))]
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
            min_errors = torch.sqrt(
                ((pos_matched - pos_true)**2).sum(-1)).mean((1, 2))
            res['error'] = min_errors.mean().cpu()
            res['std_error'] = min_errors.std().cpu()

        else:
            # return error over sequence!
            # get correctly matched sequence
            error_over_sequence = torch.sqrt(
                ((pos_matched - pos_true)**2).sum(-1)).mean(-1)
            res['error'] = error_over_sequence.mean(0).cpu()
            res['std_error'] = error_over_sequence.std(0).cpu()

        if return_velocity:
            # get velocities and transform
            vel_pred = predicted[..., 2:4]

            # get correctly matched velocities
            vel_matched = [vel_pred[i, :, permutations[j]] for i, j in selector]
            vel_matched = torch.stack(vel_matched, 0)

            # again. root of sum of squared distances per object. mean over image.
            vel_true = true[..., 2:]
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
            # way of calculating error.
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
        """Plot images of sequences and predicted bounding boxes.

        Currently not in use.
        """
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
            )

        if self.c.debug_extend_plots:
            overlap = prop_dict['overlap_ratios'].cpu().numpy()
            marg_patch = prop_dict['marginalise_flat'].detach().reshape(
                (-1, self.c.patch_width, self.c.patch_height)).cpu().numpy()

            marginalise_bg = prop_dict['marginalise_bg'].cpu().numpy()
            bg_loglik = prop_dict['bg_loglik'].cpu().numpy()

            plot_bg(marginalise_bg, bg_loglik, n_sequences=n_seq)

            patches = prop_dict['patches'].cpu().numpy()
            patches_ll = prop_dict['patches_loglik'].cpu().numpy()

            plot_patches(patches, marg_patch, overlap, patches_ll, self.c)

    def adjust_learning_rate(self, optimizer, value, step):
        """Adjust learning rate during training."""
        lr = self.c.learning_rate * np.exp(-step / value)
        lr = max(lr, self.c.min_learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def init_t(self, tensor):
        """Move tensor to self.c.device and cast to self.c.dtype."""
        return tensor.type(self.c.dtype).to(device=self.c.device)

    def train(self, num_epochs=None):
        """Run training loop.

        Also takes care of intermediate testing and logging.
        """
        print('Starting training for {}'.format(self.c.description))
        print('Only pretraining.' if self.c.supair_only else 'Full inference.')

        start_epoch = self.epoch_start
        step_counter = self.step_start

        start = time.time()
        self.test(step_counter, start)

        if num_epochs is None:
            num_epochs = self.c.num_epochs

        for epoch in range(start_epoch, num_epochs):
            for data in self.dataloader:
                now = time.time() - start
                step_counter += 1

                if self.c.debug_anneal_lr:
                    self.adjust_learning_rate(
                        self.optimizer, self.c.debug_anneal_lr, step_counter)

                # Load data
                images = self.init_t(data['present_images'])
                if self.c.action_conditioned:
                    actions = self.init_t(data['present_actions'])
                else:
                    actions = None

                # Model optimization
                self.optimizer.zero_grad()

                elbo, prop_dict, rewards = self.stove(
                    images,
                    step_counter,
                    actions,
                    self.c.supair_only)
                min_ll = -1.0 * elbo

                if self.c.action_conditioned:
                    target_rewards = data['present_rewards'][:, self.c.skip:]
                    target_rewards = self.init_t(target_rewards)
                    mse_rewards = self.reward_loss(
                        rewards.flatten(), target_rewards.flatten())

                    if self.c.debug_reward_rampup is not False:
                        reward_weight = min(
                            1, step_counter/self.c.debug_reward_rampup)
                    else:
                        reward_weight = 1

                    reward_factor = self.c.debug_reward_factor
                    min_ll = min_ll + reward_factor * reward_weight * mse_rewards
                else:
                    mse_rewards = torch.Tensor([0])

                min_ll.backward()

                if self.c.debug_gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.stove.parameters(), 1)
                self.optimizer.step()

                # Plot examples
                if step_counter % self.c.plot_every == 0:
                    plot_grad_flow(self.stove.named_parameters())
                    # full states only available after some steps
                    plot_images = images[:, self.c.skip:]

                    self.plot_results(step_counter, plot_images, prop_dict)

                # Print and log performance
                if step_counter % self.c.print_every == 0:
                    self.error_and_log(
                        elbo.item(), mse_rewards.item(), min_ll.item(),
                        prop_dict, data, step_counter, now)

                # Save parameters
                if step_counter % self.c.save_every == 0:
                    self.save(epoch, step_counter)

                if self.c.debug_test_mode and not self.c.supair_only:
                    self.save(0, 0)
                    self.test(step_counter, now)
                    break

            # Test each epoch
            if not self.c.supair_only:
                self.test(step_counter, start)

            print("Epoch: ", epoch, " finished.")

            if self.c.debug_test_mode:
                break

        # Create some more rollouts at the end of training
        if not self.c.debug_test_mode:
            for idx in self.c.rollout_idxs:
                self.long_rollout(step_counter, idx)

        # Save model in final state
        if not self.c.nolog:
            self.save(epoch, step_counter)
            succ = os.path.join(self.logger.exp_dir, 'success')
            open(succ, 'w').close()
        print('Finished Training!')

    def error_and_log(self, elbo, reward, min_ll, prop_dict, data, step_counter,
                      now, add=''):
        """Format performance metrics and pass them to logger.

        Args:
            elbo (float): Elbo value.
            reward (float): Mean reward value.
            min_ll (float): Total loss.
            prop_dict (dict): Dict from model containing further metrics.
            data (dict): Current data dict. Needed to compute errors.
            step_counter (int): Current step.
            now (int): Time elapsed.
            add (str): Identifier for log entries. Used if, e.g. this function
                is called from test() rather than train().

        """
        skip = self.c.skip

        # perf_dict contains performance values and will be passed to logger
        perf_dict = {
            'step': step_counter,
            'time': now,
            'elbo': elbo,
            'reward': reward,
            'min_ll': min_ll}

        # non z entries
        other_keys = list(filter(lambda x: x[0] != 'z', list(prop_dict.keys())))
        other_dict = {key: prop_dict[key] for key in other_keys}
        perf_dict.update(other_dict)

        # get errors for each of the z types
        z_true = data['present_labels'][:, skip:]
        z_true = self.init_t(z_true)
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
        """Evaluate performance on test data.

        Additionally
            - tests performance of generative model, i.e. rollout performance,
            - creates a rollout gif.

        Args:
            step_counter (int): Current step.
            start (int): Time at beginning of training

        """
        self.stove.eval()
        skip = self.c.skip
        for i, data in enumerate(self.test_dataloader, 0):
            now = time.time() - start

            # Load Data
            present = self.init_t(data['present_images'])
            if self.c.action_conditioned:
                actions = self.init_t(data['present_actions'])
                future_actions = self.init_t(data['future_actions'])
                future_rewards = self.init_t(data['future_rewards'])
            else:
                actions, future_actions, future_rewards = None, None, None

            # Propagate through model
            elbo, prop_dict, rewards = self.stove(
                present,
                self.c.plot_every,
                actions,
                self.c.supair_only
                )

            min_ll = -1.0 * elbo

            if self.c.action_conditioned:
                target_rewards = self.init_t(data['present_rewards'][:, skip:])
                mse_rewards = self.reward_loss(
                    rewards.flatten(), target_rewards.flatten())

                if self.c.debug_reward_rampup is not False:
                    reward_weight = min(
                        1, step_counter/self.c.debug_reward_rampup)
                else:
                    reward_weight = 1

                reward_factor = self.c.debug_reward_factor
                min_ll = min_ll + reward_factor * reward_weight * mse_rewards
            else:
                mse_rewards = torch.Tensor([0])

            # Log Errors
            self.error_and_log(
                elbo.item(), mse_rewards.item(), min_ll.item(), prop_dict, data,
                step_counter, now, add='_roll')

            if self.c.debug_core_appearance:
                appearances = prop_dict['obj_appearances'][:, -1]
            else:
                appearances = None

            z_pred, rewards_pred = self.stove.rollout(
                prop_dict['z'][:, -1], actions=future_actions,
                appearance=appearances)

            if self.c.action_conditioned:
                future_reward_loss = self.reward_loss(
                    rewards_pred.flatten(), future_rewards.flatten())
            else:
                future_reward_loss = 0

            z_true = self.init_t(data['future_labels'])
            error_dict = self.prediction_error(
                z_pred[..., 2:], z_true)

            perf_dict = {
                'step': step_counter, 'time': now, 'elbo': elbo,
                'reward': future_reward_loss}

            perf_dict.update(error_dict)
            other_keys = list(filter(lambda x: x[0] != 'z', list(prop_dict.keys())))
            other_dict = {key: prop_dict[key] for key in other_keys}
            perf_dict.update(other_dict)
            perf_dict['type'] = 'rollout'

            self.logger.performance(perf_dict)

            if self.c.debug_test_mode:
                break
            if i > 7:
                break

        self.long_rollout(step_counter)
        self.stove.train()

    @torch.no_grad()
    def long_rollout(self, step_counter, idx=0, with_logging=True, actions=None):
        """Create one long rollout and save it as an animated GIF.

        Args:
            step_counter (int): Current step.
            idx (int): Index of sequence in test data set.
            with_logging (int): Also save errors over sequences in csv for
                rollout plots.
            actions (n, T): Pass actions different from those in the test
                set to see if model has understood actions.

        """
        self.stove.eval()

        step = self.c.frame_step
        visible = self.c.num_visible
        batch_size = self.c.batch_size
        skip = self.c.skip
        long_rollout_length = self.c.num_frames // step - visible

        np_total_images = self.test_dataset.total_img
        np_total_labels = self.test_dataset.total_data
        # apply step and batch size once
        total_images = self.init_t(torch.tensor(
            np_total_images[:batch_size, ::step]))
        total_labels = self.init_t(torch.tensor(
            np_total_labels[:batch_size, ::step]))

        if self.c.action_conditioned:
            if actions is None:
                total_actions = self.test_dataset.total_actions
            else:
                total_actions = actions

            total_actions = self.init_t(torch.tensor(
                total_actions[:batch_size, ::step]))
            action_input = total_actions[:, :visible]

            total_rewards = self.test_dataset.total_rewards
            total_rewards = total_rewards[:batch_size, ::step]
            real_rewards = total_rewards[idx, self.c.skip:, 0]

            # need some actions for rollout
            true_future_actions = total_actions[:, visible:(visible+long_rollout_length)]
            action_recon = total_actions[idx:idx+2, :(visible+long_rollout_length)]

        else:
            action_input = None
            true_future_actions = None
            action_recon = None

        # first obtain reconstruction of input.
        stove_input = total_images[:, :visible]
        _, prop_dict2, rewards_recon = self.stove(
            stove_input, self.c.plot_every, action_input)
        z_recon = prop_dict2['z']

        # use last state to do rollout
        if self.c.debug_core_appearance:
            appearances = prop_dict2['obj_appearances'][:, -1]
        else:
            appearances = None

        z_pred, rewards_pred = self.stove.rollout(
            z_recon[:, -1], num=long_rollout_length, actions=true_future_actions,
            appearance=appearances)

        # assemble complete sequences as concat of reconstruction and prediction
        simu_recon = z_recon.detach()
        simu_rollout = z_pred.detach()
        simu = torch.cat([simu_recon, simu_rollout], 1)

        if not self.c.nolog and with_logging:
            # get prediction error over long sequence for loogging
            real_labels = total_labels[:, skip:(visible+long_rollout_length)]
            error_dict = self.prediction_error(
                simu[..., 2:6], real_labels, return_velocity=True, return_full=True,
                return_id_swaps=False)

            for name, data in error_dict.items():
                file = os.path.join(self.logger.exp_dir, '{}.csv'.format(name))
                with open(file, 'a') as f:
                    f.write(','.join(['{:.6f}'.format(i) for i in data])+'\n')

        # only select positions and translate to [0, 10] frame
        simu = simu[idx, :, :, 2:4].detach().cpu().numpy()

        # also get a reconstruction of z along entire sequence
        stove_input = total_images[idx:idx+2, :(visible+long_rollout_length)]
        elbo, prop_dict3, recon_reward = self.stove(
            stove_input, self.c.plot_every, actions=action_recon)
        recon = prop_dict3['z'][0, :, :, 2:4]
        recon = recon.cpu().numpy()
        recon_reward = recon_reward.cpu().numpy()[0].squeeze()

        real = np_total_labels[idx, self.c.skip:, :, :2]

        if self.c.action_conditioned:
            # add rewards to gif
            rewards_model = torch.cat([rewards_recon, rewards_pred], 1).squeeze()[idx]
            rewards_model = rewards_model.detach().cpu().numpy()

        # Make Gifs
        gif_path = os.path.join(self.logger.exp_dir, 'gifs')

        print("Make GIFs in {}".format(gif_path))

        gifs = [real, simu, simu, recon, recon]
        if self.c.action_conditioned:
            rewards = [real_rewards, rewards_model, rewards_model,
                       recon_reward, recon_reward]
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
        # Set stove to train mode again.
        self.stove.train()
