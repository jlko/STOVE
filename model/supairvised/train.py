"""Contains trainer for supairvised baseline."""
import os
import itertools
import time
import numpy as np
import torch
from torch import nn

from ..video_prediction.train import AbstractTrainer
from ..video_prediction.load_data import StoveDataset
from ..utils.visualize import animate, color
from ..utils.utils import ExperimentLogger

class SupTrainer(AbstractTrainer):
    """Trainer for dynamics prediction in supairvised or supervised scenarios.

    Supairvised: State supervision is given by SuPAIR.
    Supervised: Ground truth states from environment are taken.

    May not support all features of Stove trainer.
    """

    def __init__(self, config, stove, train_dataset, test_dataset):
        """Set up trainer."""
        super().__init__(config, stove, train_dataset, test_dataset)

         # types are: vin-true, vin-sup, sup-true
         # frame is: recon, rollout, total
        attributes = \
            ['step', 'time',
             'error', 'type', 'frame', 'test']
        log_str = '{:d},' + 2*'{:.5f},' + 2*'{},' + '{}\n'
        self.logger = ExperimentLogger(config, attributes, log_str)

    def match_positions(self, true, supair):
        """Match position ordering of true with supair.
        
        Args:
            true, supair (torch.Tensor): True object and inferred object states.

        Returns:
            matched_pred (torch.Tensor): Assign most likely ordering of supair
                ordering as the object permutation with lowest RMSE over image.

        """
        # only match based on positions, do on per image (nT) basis
        pos_true = true[..., :2]
        pos_pred = supair[..., :2]

        errors = []
        permutations = list(itertools.permutations(range(0, self.c.num_obj)))

        # change to assign just one ordering per sequence
        T = min(4, pos_pred.shape[1])

        for perm in permutations:
            error = ((pos_pred[:, :T, perm] - pos_true[:, :T])**2)
            error = torch.sqrt(error.sum(-1)).mean((1, 2))
            errors += [error]
            # sum_k/T(sum_j/o(root(sum_i((x_i0-x_i1)**2))))
            # sum_i over x and y coordinates -> root(sum squared) is distance of
            # objects for that permutation.
            # assign min per image

        # shape (n, T, o!)
        errors = torch.stack(errors, 1)
        # sum to get error per image
        _, idx = errors.min(1)
        # idx now contains a winning permutation per image!
        selector = list(zip(range(idx.shape[0]), idx.cpu().tolist()))

        matched_pred = [supair[i, :, permutations[j]] for i, j in selector]
        matched_pred = torch.stack(matched_pred, 0)

        return matched_pred

    def compute_loss(self, present_labels, future_labels, recons, preds):
        """Compute loss for reconstruction and prediction."""
        
        delta_xy_labels = future_labels[:, 1:] - future_labels[:, :-1]
        delta_xy_preds = preds[:, 1:] - preds[:, :-1]

        loss = nn.MSELoss()
        df = self.c.discount_factor
        pred_loss = 0.0
        delta_pred_loss = 0.0

        end = 2 if self.c.debug_disable_v_error else 4

        for delta_t in range(0, self.c.num_rollout):
            pred_loss += (df ** (delta_t + 1)) * \
                    loss(preds[:, delta_t, :, :end],
                         future_labels[:, delta_t, :, :end])

            if (not self.c.debug_disable_v_diff_error and
                    (delta_t < self.c.num_rollout - 1)):
                delta_pred_loss += (df ** (delta_t + 1)) * \
                     6 * loss(delta_xy_preds[:, delta_t, :, :end],
                              delta_xy_labels[:, delta_t, :, :end])

        # make pred and recon loss comparable. pred_loss was scaling with
        # num_rollout (does not really matter omptimisation wise, since recon
        # loss tends to 0 fast, or is 0 in supervised case)
        # pred_loss = pred_loss / self.c.num_rollout
        recon_loss = loss(recons[..., :end], present_labels[..., :end])
        total_loss = pred_loss + recon_loss + delta_pred_loss

        return total_loss, pred_loss, recon_loss

    def prediction_error(self, predicted, real, suffix=''):
        """Log prediction errors over sequence for rollout error."""
        if suffix != '' and suffix[0] != '_':
            suffix = '_' + suffix

        # only do positions
        predicted = predicted[..., :2]
        real = real[..., :2]

        # mean error over sequence
        tmp = torch.sqrt(((predicted - real)**2).sum(-1)).mean(-1)
        error = dict()
        error['error'] = tmp.mean(0)
        error['std_error'] = tmp.std(0)

        for name, data in error.items():
            file = os.path.join(self.logger.exp_dir, '{}{}.csv'.format(name, suffix))
            with open(file, 'a') as f:
                f.write(','.join(['{:.6f}'.format(i) for i in data])+'\n')

    @staticmethod
    def its(states):
        """Inverse transform of object positions to -1, 1 frame for plotting."""
        
        # transform back to [0, 10]
        pos = states[..., :2] * 5
        vel = states[..., 2:4] / 2

        # transform to [-1, 1]
        pos = pos / 10 * 2 - 1
        vel = vel / 10 * 2

        return torch.cat([pos, vel], -1)

    def train(self):
        """Execute model training."""
        print('Using supair!' if self.c.load_encoder else 'Supervised!')
        step_counter = 0
        num_rollout = self.c.num_rollout
        start = time.time()
        self.test(step_counter, start)

        for epoch in range(self.c.num_epochs):
            for i, data in enumerate(self.dataloader, 0):
                now = time.time() - start

                step_counter += 1
                present_images, future_images, future_labels, present_labels = \
                    data['present_images'], data['future_images'], \
                    data['future_labels'], data['present_labels']

                present_images = present_images.to(self.c.device)
                future_images = future_images.to(self.c.device)
                present_labels = present_labels.to(self.c.device)
                future_labels = future_labels.to(self.c.device)

                if self.c.load_encoder:
                    all_images = torch.cat([present_images, future_images], 1)
                    sup_states = self.stove.sup(all_images).detach()

                    # match supair states object ordering to true order
                    true_states = torch.cat(
                        [present_labels[:, 1:], future_labels], 1)
                    sup_states = self.match_positions(true_states, sup_states)

                    sup_present = sup_states[:, :self.c.num_visible-1]
                    sup_future = sup_states[:, self.c.num_visible-1:]

                    input = sup_present
                    future = sup_future
                    log_type = 'vin_sup'

                else:
                    input = present_labels
                    future = future_labels
                    log_type = 'vin_true'

                self.optimizer.zero_grad()

                state_pred, state_recon = self.stove.dyn(
                   input,
                   num_rollout=num_rollout,
                   debug_rollout=self.c.debug_rollout_training)

                # true against prediction (from VIN)
                total_loss, pred_loss, recon_loss = \
                    self.compute_loss(input, future,
                                      state_recon, state_pred)

                total_loss.backward()
                self.optimizer.step()

                if step_counter % self.c.print_every == 0:
                    perf_dict = dict()
                    perf_dict['step'] = step_counter
                    perf_dict['time'] = now

                    self.error_and_log(
                        perf_dict, total_loss.item(), pred_loss.item(),
                        recon_loss.item(), log_type)

                # also document aux losses
                if step_counter % self.c.print_every == 0 and self.c.load_encoder:

                    # supair against true labels
                    rv_present = present_labels[:, 1:]
                    rv_future = future_labels
                    sup_total_loss, sup_pred_loss, sup_recon_loss = \
                        self.compute_loss(
                            rv_present, rv_future,
                            sup_present, sup_future)

                    # also document loss VIN against real labels
                    vin_total_loss, vin_pred_loss, vin_recon_loss = \
                        self.compute_loss(
                            rv_present, rv_future,
                            state_recon, state_pred)

                    # for supair, total should be eq to pred and recon.
                    self.error_and_log(
                        perf_dict, sup_total_loss.item(), sup_pred_loss.item(),
                        sup_recon_loss.item(), 'sup_true')

                    self.error_and_log(
                        perf_dict, vin_total_loss.item(), vin_pred_loss.item(),
                        vin_recon_loss.item(), 'vin_true')

                if self.c.debug_test_mode:
                    # break out of steps
                    break

                if step_counter % self.c.save_every == 0:
                    self.save(epoch, step_counter)

            if self.c.debug_test_mode:
                # break out of epochs
                break

            self.test(step_counter, start)
            print("epoch ", epoch, " Finished")

        self.save(epoch, step_counter)
        print('Finished Training')


    def error_and_log(self, perf_dict, total_loss, pred_loss, recon_loss,
                      log_type, test=False):
        """Interface with experiment_logger to log errors.

        In supairvised scenario, there are different errors to consider.
        1. Reconstruction errors of observed states and errors of predicted 
            states.
        2. Errors of model against supair and of supair against true states.
        """
        frames = ['total', 'pred', 'recon']
        errors = [total_loss, pred_loss, recon_loss]

        for error, frame in zip(errors, frames):
            perf_dict['error'] = error
            perf_dict['type'] = log_type
            perf_dict['frame'] = frame
            perf_dict['test'] = str(test)
            self.logger.performance(perf_dict)

    @torch.no_grad()
    def test(self, step_counter, start):
        """Test model performance on test set."""
        self.stove.eval()
        now = time.time() - start
        perf_dict = {'step': step_counter, 'time': now}

        for i, data in enumerate(self.test_dataloader, 0):
            present_images, future_images, present_labels, future_labels,  = \
                data['present_images'], data['future_images'],\
                data['present_labels'], data['future_labels']

            present_images = present_images.to(self.c.device)
            future_images = future_images.to(self.c.device)
            present_labels = present_labels.to(self.c.device)
            future_labels = future_labels.to(self.c.device)

            if self.c.load_encoder:
                all_images = torch.cat([present_images, future_images], 1)
                sup_states = self.stove.sup(all_images).detach()
                
                # match supair states object ordering to true order
                true_states = torch.cat(
                    [present_labels[:, 1:], future_labels], 1)
                sup_states = self.match_positions(true_states, sup_states)

                sup_present = sup_states[:, :self.c.num_visible-1]
                sup_future = sup_states[:, self.c.num_visible-1:]

                input = sup_present
                future = sup_future
                offset = 1

            else:
                input = present_labels
                future = future_labels
                offset = 0

            state_pred, state_recon = self.stove.dyn(
                input,
                num_rollout=self.c.num_rollout,
                debug_rollout=self.c.debug_rollout)

            # as total test loss, we now want to measure against true states
            vin_total_loss, vin_pred_loss, vin_recon_loss = \
                self.compute_loss(
                    present_labels[:, offset:], future_labels,
                    state_recon, state_pred)

            self.error_and_log(
                perf_dict, vin_total_loss.item(), vin_pred_loss.item(),
                vin_recon_loss.item(), 'vin_true', test=True)

            if self.c.debug_test_mode:
                break

            if i > 7:
                break

        self.long_rollout(step_counter)

    @torch.no_grad()
    def long_rollout(self, step_counter, idx=0, with_logging=True):
        """Create long rollouts save as gif and log their mse rollout error."""

        total_images = self.test_dataset.total_img
        total_labels = self.test_dataset.total_data
        step = self.c.frame_step
        visible = self.c.num_visible
        batch_size = self.c.batch_size

        long_rollout_length = self.c.num_frames // step - visible

        true_states = total_labels[:batch_size, :visible*step:step]
        true_states = torch.tensor(true_states).double().to(self.c.device)

        if self.c.load_encoder:
            sup_input = total_images[:batch_size, :visible*step:step]
            sup_input = torch.tensor(sup_input).to(self.c.device)
            sup_states = self.stove.sup(sup_input).detach()
            # account for zeros in sup_states
            sup_states = self.match_positions(true_states[:, 1*step:], sup_states)
            input = sup_states
        else:
            input = true_states

        pred, recon = self.stove.dyn(input, long_rollout_length,
                               debug_rollout=self.c.debug_rollout
                               )

        simu_rollout = pred.detach()
        simu_recon = recon.detach()
        simu = torch.cat([simu_recon, simu_rollout], 1)

        if not self.c.nolog and with_logging:
            # get prediction error over long sequence for loogging
            offset = 1 if self.c.load_encoder else 0
            real_labels = torch.Tensor(total_labels, device=self.c.device)
            real_labels = real_labels.type(self.c.dtype)
            real_labels = real_labels[
                :batch_size, offset:(visible+long_rollout_length), :, :2]
            # simu should already be matched to real_labels bc input was aligned
            self.prediction_error(simu, real_labels, suffix='')

        print("Making GIFs.")

        # Saving
        # Rescale to 0, 10 frame
        simu_s = self.its(simu[idx, :, :, :2]).cpu().numpy()
        plot_labels = self.its(torch.from_numpy(total_labels[idx, 1::step]))
        plot_labels = plot_labels.numpy()

        if not self.c.nolog:
            gif_path = os.path.join(self.logger.exp_dir, 'gifs')
        else:
            gif_path = os.path.join(self.c.experiment_dir, 'tmp')

        res = self.c.height
        r = self.c.r

        animate(plot_labels, gif_path, 'real_{:02d}'.format(idx))
        animate(simu_s, gif_path, 'rollout_{:02d}_{:05d}'.format(
            step_counter, idx), res=res, r=r)
        animate(simu_s, gif_path, 'rollout_{:02d}'.format(idx),
                 res=res, r=r,)

        # also make rollouts of how supair would have seen the whole sequence
        if self.c.load_encoder:   

            sup_input = total_images[idx:idx+1, ::step]
            sup_input = torch.tensor(sup_input).to(self.c.device)
            sup_states = self.stove.sup(sup_input).detach()
            true = torch.tensor(plot_labels).to(self.c.device)
            true = true.double().unsqueeze(0)
            sup_states = self.match_positions(true, sup_states)
            sup_fixed = self.stove.sup.fix_supair(sup_states)
            if not self.c.nolog and with_logging:
                self.prediction_error(simu, sup_fixed, suffix='sup')

            sup_states = self.its(sup_states)
            animate(sup_states[0].detach().cpu().numpy(), gif_path,
                    'supair_{:02d}'.format(idx), res=res, r=r)

        print("Done")
