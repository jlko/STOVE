"""Contains supervised version of dynamics model."""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from ..video_prediction.dynamics import Dynamics


class SupervisedDynamics(Dynamics):
    """Graph Neural Network Dynamics Model.

    Modified variant for supervised prediction.
    """

    def __init__(self, config):
        """Set up model.

        Difference to full dynamics core. Now,
        """
        enc_input_size = 4
        super().__init__(config, enc_input_size)
        self.c = config

    def forward(self, x, num_rollout=8, debug_rollout=False):
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
        if debug_rollout:
            # initialize latent space in by applying core to observed states
            cur_state_code = self.state_enc(x[:, 0])
            for i in range(1, x.shape[1]):
                result, _ = self.core(cur_state_code, 0)
                result = torch.cat(
                    [result[..., :2] + cur_state_code[..., :2],
                     result[..., 2:]], -1)

                cur_state_code = torch.cat(
                    [x[:, i, :, :4], result[..., 4:]], -1)

        else:
            state_codes = self.state_enc(x)
            state_codes[..., :4] = x
            # the 4 state codes (n, o, cl)
            cur_state_code = state_codes[:, -1]

        rollouts = []
        for _ in range(num_rollout):
            result, _ = self.core(cur_state_code, 0)
            result = torch.cat(
                [result[..., :2] + cur_state_code[..., :2],
                 result[..., 2:]], -1)

            rollouts.append(result)
            cur_state_code = result

        rollouts = torch.stack(rollouts, 1)
        rollout_states = rollouts[..., :4]

        return rollout_states, x
