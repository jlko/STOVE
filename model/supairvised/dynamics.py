"""Contains supervised version of dynamics model."""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..video_prediction.dynamics import Dynamics


class SupervisedDynamics(Dynamics):
    """Graph Neural Network Dynamics Model.

    Modified variant for supervised prediction.
    """

    def __init__(self, config):
        """Set up model.

        Difference to full dynamics core. Now,
        """
        enc_input_size = 16
        super().__init__(config, enc_input_size)
        self.c = config
        self.latent_prior = Normal(
            torch.Tensor([0], device=self.c.device),
            torch.Tensor([1], device=self.c.device))

        # fake object appearances
        if self.c.debug_core_appearance:
            apps = torch.tensor([[.5, 0, 0], [0, .5, 0], [0, 0, .5]])
            self.appearances = apps.type(self.c.dtype).to(device=self.c.device)

    def super_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, x, num_rollout=8, debug_rollout=False, actions=None):
        """Rollout a given sequence of observations using the model.

        Args:
            x (torch.Tensor): The given sequence of states of shape
                (n, 4, o, 4).
            num_rollout (int): The number of future states to be predicted.

        Returns:
            rollout_states (torch.Tensor): Predicted future states
                of shape (n, num_rollout, o, 4).

        """
        # get high-dimensional encoding of observed states
        # keep positions and velocities
        if actions is None:
            actions = [None]
            action_len = 1
        else:
            actions = actions.transpose(0, 1)
            action_len = actions.shape[0]

        # for long rollouts we might not have ground truth actions
        # just repeat actions

        if self.c.debug_core_appearance:
            appearances = self.appearances.unsqueeze(0).repeat(x.shape[0], 1, 1)
        else:
            appearances = None

        if debug_rollout:
            # initialize latent space in by applying core to observed states
            prior_shape = (x.shape[0], x.shape[2], self.c.cl - 4)
            cur_state_code = torch.cat(
                [x[:, 0], self.latent_prior.rsample(prior_shape).squeeze()], -1)
            for i in range(1, x.shape[1]):
                result, _ = self.super_forward(
                    cur_state_code, 0, actions=actions[i % action_len],
                    obj_appearances=appearances, lim_enc=4)
                result = torch.cat(
                    [result[..., :2] + cur_state_code[..., :2],
                     result[..., 2:]], -1)

                # for observed states, put xv explicitly in latent
                cur_state_code = torch.cat(
                    [x[:, i, :, :4], result[..., 4:]], -1)

        else:
            state_codes = self.state_enc(x)
            state_codes[..., :4] = x
            # the 4 state codes (n, o, cl)
            cur_state_code = state_codes[:, -1]

        rollouts = []
        for i in range(x.shape[1], x.shape[1]+num_rollout):
            result, _ = self.super_forward(
                cur_state_code, 0, actions=actions[i % action_len],
                obj_appearances=appearances, lim_enc=4)
            result = torch.cat(
                [result[..., :2] + cur_state_code[..., :2],
                 result[..., 2:]], -1)

            rollouts.append(result)
            cur_state_code = result

        rollouts = torch.stack(rollouts, 1)
        rollout_states = rollouts[..., :4]

        return rollout_states, x
