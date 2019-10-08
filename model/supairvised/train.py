# THINGS LEFT TO DO
# ROLLOUT over time error in test
# change scale to be the same

import os
import itertools
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from load_data import VinDataset
from visualize import plot_positions, animate, color
from utils import ExperimentLogger


class Trainer:
    def __init__(self, config, net, supervisor):
        self.net = net
        self.supervisor = supervisor

        self.params = net.parameters()
        self.initial_values = {}
        self.c = config


        attributes = \
            ['step', 'time',
             'error', 'type', 'frame', 'test']

         # types are: vin-true, vin-sup, sup-true
         # frame is: recon, rollout, total
        log_str = '{:d},' + 2*'{:.5f},' + 2*'{},' + '{}\n'

        self.logger = ExperimentLogger(config, attributes, log_str)

        train_dataset = VinDataset(self.c)
        self.dataloader = DataLoader(train_dataset,
                                     batch_size=self.c.batch_size,
                                     shuffle=True,
                                     num_workers=self.c.num_workers,
                                     drop_last=True)

        self.test_dataset = VinDataset(self.c,
                                       test=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.c.batch_size,
                                          shuffle=True,
                                          num_workers=self.c.num_workers,
                                          drop_last=True)

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0005)

        if config.supair_path is not None:
            self.load_weights(config.supair_path)
            self.disable_gradient()

        if config.checkpoint_path is not None:
            self.load()


    def load_weights(self, path):
        """Load Attention RNN weights from pretrained SuPAIR."""
        pretrained_dict = torch.load(path, map_location=self.c.device)
        model_dict = self.supervisor.state_dict()
        # 1. filter out unnecessary keys, only load encoder weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.supervisor.load_state_dict(model_dict)
        print('Successfully loaded the following supair parameters from checkpoint: {}.'.format(path))
        print(pretrained_dict.keys())

    def disable_gradient(self):
        """Disable gradients for encoder."""
        for p in self.supervisor.parameters():
            p.requires_grad = False
        print('Disabled gradients for encoder')

    def save(self):
        torch.save(self.net.state_dict(), os.path.join(
                            self.logger.exp_dir, "checkpoint"))
        print('Parameters for net saved')

    def load(self):
        self.net.load_state_dict(torch.load(
            self.c.checkpoint_path, map_location=self.c.device))
        print('Parameters for net loaded')

    def match_positions(self, true, supair):
        """Match position ordering of true with supair.

        Assign most likely ordering as the permutation with lowest RMSE over image.

        :returns matched_pred: Return permutation of supair labelling that best matches true labels.

        """
        # only match based on positions, do on per image (nT) basis
        pos_true = true[..., :2]
        pos_pred = supair[..., :2]

        errors = []
        permutations = list(itertools.permutations(range(0, self.c.num_obj)))

        # change to assign just one ordering per sequence
        T = min(4, pos_pred.shape[1])

        for perm in permutations:
            errors += [torch.sqrt(((pos_pred[:, :T, perm] - pos_true[:, :T])**2).sum(-1)).mean((1, 2))]
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
        # loss *only* in positions. this is what worked best.
        # also has convenience that I can equate loss and eror an compare
        # w.r.t. other code
        delta_xy_labels = future_labels[:, 1:] - future_labels[:, :-1]
        delta_xy_preds = preds[:, 1:] - preds[:, :-1]

        loss = nn.MSELoss()
        df = self.c.discount_factor
        pred_loss = 0.0
        delta_pred_loss = 0.0

        end = 4 if not self.c.debug_disable_v_error else 2
        for delta_t in range(0, self.c.num_rollout):
            pred_loss += (df ** (delta_t + 1)) * \
                    loss(preds[:, delta_t, :, :end], future_labels[:, delta_t, :, :end])

            if not self.c.debug_disable_v_diff_error and (delta_t < self.c.num_rollout - 1):
                delta_pred_loss += (df ** (delta_t + 1)) * \
                     6 * loss(delta_xy_preds[:, delta_t, :, :end], delta_xy_labels[:, delta_t, :, :end])

        # make pred and recon loss comparable. pred_loss was scaling with num_rollout
        # (does not really matter omptimisation wise, since recon loss tended to 0 fast)
        # DEBUG. lose normalisation
        # pred_loss = pred_loss / self.c.num_rollout
        recon_loss = loss(recons[..., :end], present_labels[..., :end])
        total_loss = pred_loss + recon_loss + delta_pred_loss

        return total_loss, pred_loss, recon_loss

    def prediction_error(self, predicted, real, suffix=''):
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
    def rv(states, factor=1):
        """Supair coordinates are in [-1, 1], velocities are in [-0.2, 0.2]
            Reweigh velocities with a factor of 5 for balanced MSE.
            Sorry for bad function naming. RV, reweigh velocities.
        """
        #return torch.cat([states[..., :2], factor * states[..., 2:4]], -1)
        return states

    @staticmethod
    def ts(states):
        """Supair coordinates are in [-1, 1], velocities are in [-0.2, 0.2]
           Original data coordinates are in [0, 10] transform coordinates
           and velocities to Supair frame s.t. losses are consistent.
        """
        # this used to be in dataloader, now that we use global loader put it here
        # form 0,10 to 0,1 to 0,2 to -1,1
        # pos = states[..., :2] / 10
        # # velocities are diffs of positions, -1 cancels
        # vel = states[..., 2:4] * 5

        # return torch.cat([pos, vel], -1)
        return states

    @staticmethod
    def its(states):
        """ Inverse transform for plotting.
        """
        # pos = (states[..., :2] + 1)/2*10
        # # velocities are diffs of positions, -1 cancels
        # vel = states[..., 2:4]/2*10

        pos = states[..., :2] * 5
        # velocities are diffs of positions, -1 cancels
        vel = states[..., 2:4] / 2

        return torch.cat([pos, vel], -1)
        # return states


    def train(self):
        print('Using supair!' if self.c.use_supair else 'Supervised!')
        step_counter = 0
        num_rollout = self.c.num_rollout
        start = time.time()
        self.test(step_counter, start)

        for epoch in range(self.c.num_epochs):
            for i, data in enumerate(self.dataloader, 0):
                now = time.time() - start

                step_counter += 1
                present_images, future_images, future_labels, present_labels = \
                    data['present_images'], data['future_images'], data['future_labels'], data['present_labels']

                present_images = present_images.to(self.c.device)
                future_images = future_images.to(self.c.device)
                present_labels = self.ts(present_labels.to(self.c.device))
                future_labels = self.ts(future_labels.to(self.c.device))

                if self.c.use_supair:
                    all_images = torch.cat([present_images, future_images], 1)
                    sup_states = self.supervisor(all_images).detach()

                    # match supair states object ordering to true order
                    true_states = torch.cat([present_labels[:, 1:], future_labels], 1)
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

                state_pred, state_recon = self.net(input,
                                                   num_rollout=num_rollout,
                                                   visual=self.c.visual,
                                                   debug_rollout=self.c.debug_rollout_training)

                # true against prediction (from VIN)
                total_loss, pred_loss, recon_loss = \
                    self.compute_loss(self.rv(input), self.rv(future),
                                      self.rv(state_recon), self.rv(state_pred))

                total_loss.backward()
                # disable grad norm
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optimizer.step()

                if step_counter % self.c.print_every == 0:
                    perf_dict = dict()
                    perf_dict['step'] = step_counter
                    perf_dict['time'] = now

                    self.error_and_log(
                        perf_dict, total_loss.item(), pred_loss.item(),
                        recon_loss.item(), log_type)

                # also document aux losses
                if step_counter % self.c.print_every == 0 and self.c.use_supair:

                    # supair against true labels
                    rv_present = self.rv(present_labels[:, 1:])
                    rv_future = self.rv(future_labels)
                    sup_total_loss, sup_pred_loss, sup_recon_loss = \
                        self.compute_loss(rv_present, rv_future,
                                          self.rv(sup_present), self.rv(sup_future))

                    # also document loss VIN against real labels
                    vin_total_loss, vin_pred_loss, vin_recon_loss = \
                        self.compute_loss(rv_present, rv_future,
                                          self.rv(state_recon), self.rv(state_pred))

                    # for supair, total should be eq to pred and recon.
                    self.error_and_log(
                        perf_dict, sup_total_loss.item(), sup_pred_loss.item(),
                        sup_recon_loss.item(), 'sup_true')

                    self.error_and_log(
                        perf_dict, vin_total_loss.item(), vin_pred_loss.item(),
                        vin_recon_loss.item(), 'vin_true')

                # Draw example
                if step_counter % self.c.plot_every == 0:
                    real = torch.cat([present_labels[0], future_labels[0]]).cpu().numpy()
                    simu = torch.cat([state_recon[0], state_pred[0]]).detach().cpu().numpy()
                    plot_positions(real, self.c.img_folder, 'real')
                    plot_positions(simu, self.c.img_folder, 'rollout')

                if self.c.debug_test_mode:
                    # break out of steps
                    break

                if step_counter % self.c.save_every == 0:
                    self.save()

            if self.c.debug_test_mode:
                # break out of epochs
                break

            self.test(step_counter, start)
            print("epoch ", epoch, " Finished")

        self.save()
        print('Finished Training')


    def error_and_log(self, perf_dict, total_loss, pred_loss, recon_loss, log_type, test=False):
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
        self.net.eval()
        now = time.time() - start
        perf_dict = {'step': step_counter, 'time': now}

        for i, data in enumerate(self.test_dataloader, 0):
            present_images, future_images, present_labels, future_labels,  = \
                data['present_images'], data['future_images'], data['present_labels'], data['future_labels']

            present_images = present_images.to(self.c.device)
            future_images = future_images.to(self.c.device)
            present_labels = self.ts(present_labels.to(self.c.device))
            future_labels = self.ts(future_labels.to(self.c.device))

            if self.c.use_supair:
                all_images = torch.cat([present_images, future_images], 1)
                sup_states = self.supervisor(all_images).detach()
                # match supair states object ordering to true order
                true_states = torch.cat([present_labels[:, 1:], future_labels], 1)
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

            state_pred, state_recon = self.net(
                input,
                num_rollout=self.c.num_rollout,
                visual=self.c.visual,
                debug_rollout=self.c.debug_rollout)

            # as total test loss, we now want to measure against true states
            vin_total_loss, vin_pred_loss, vin_recon_loss = \
                self.compute_loss(self.rv(present_labels[:, offset:]), self.rv(future_labels),
                                  self.rv(state_recon), self.rv(state_pred))

            self.error_and_log(
                perf_dict, vin_total_loss.item(), vin_pred_loss.item(),
                vin_recon_loss.item(), 'vin_true', test=True)

            if self.c.debug_test_mode:
                break

            if i > 7:
                break

        self.long_rollout(step_counter)

    def long_rollout(self, step_counter, idx=0, with_logging=True):
        # todo: give me error of rollout against true and (if with supair): error
        # against rollout on true

        # Create one long rollout and save it as an animated GIF
        total_images = self.test_dataset.total_img
        total_labels = self.test_dataset.total_data
        step = self.c.frame_step
        visible = self.c.num_visible
        batch_size = self.c.batch_size

        long_rollout_length = self.c.num_frames // step - visible

        true_states = torch.tensor(total_labels[:batch_size, :visible*step:step]).double().to(self.c.device)
        true_states = self.ts(true_states)

        if self.c.use_supair:
            sup_states = self.supervisor(torch.tensor(total_images[:batch_size, :visible*step:step]).to(self.c.device)).detach()
            # account for zeros in sup_states
            sup_states = self.match_positions(true_states[:, 1*step:], sup_states)
            input = sup_states
        else:
            input = true_states

        pred, recon = self.net(input, long_rollout_length,
                               visual=self.c.visual,
                               debug_rollout=self.c.debug_rollout
                               )

        simu_rollout = pred.detach()
        simu_recon = recon.detach()
        simu = torch.cat([simu_recon, simu_rollout], 1)

        if not self.c.nolog and with_logging:
            # Get prediction error over long sequence for loogging
            offset = 1 if self.c.use_supair else 0
            real_labels = torch.Tensor(total_labels, device=self.c.device).type(self.c.dtype)
            real_labels = real_labels[:batch_size, offset:(visible+long_rollout_length), :, :2]
            # simu should already be matched to real_labels bc input was aligned
            self.prediction_error(simu, real_labels, suffix='')

        print("Make GIFs")

        # Saving
        # Rescale to 0, 10 frame
        simu_s = self.its(simu[idx, :, :, :2]).cpu().numpy()
        total_labels = self.its(torch.from_numpy(total_labels[idx, 1::step])).numpy()

        if not self.c.nolog:
            gif_path = os.path.join(self.logger.exp_dir, 'gifs')
        else:
            gif_path = os.path.join(self.c.experiment_dir, 'tmp')

        animate(total_labels, gif_path, 'real_{:02d}'.format(idx))
        animate(simu_s, gif_path, 'rollout_{:02d}_{:05d}'.format(step_counter, idx))
        animate(simu_s, gif_path, 'rollout_{:02d}'.format(idx), res=total_images.shape[-1], r=0.7)
        # also make rollouts of how supair would have seen the whole sequence
        if self.c.use_supair:

            sup_states = self.supervisor(torch.tensor(total_images[idx:idx+1, ::step]).to(self.c.device)).detach()
            true = torch.tensor(total_labels).to(self.c.device).double().unsqueeze(0)
            sup_states = self.match_positions(true, sup_states)

            if not self.c.nolog and with_logging:
                self.prediction_error(simu, sup_fixed, suffix='sup')

            sup_states = self.its(sup_states)

            animate(sup_states[0].detach().cpu().numpy(), gif_path, 'supair_{:02d}'.format(idx))


        print("Done")
        self.net.train()
