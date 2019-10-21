"""LSTM encoder for SuPAIR."""

import torch
import torch.nn as nn


class RnnStates(nn.Module):
    """Get object states from RNN."""

    def __init__(self, config):
        """Set up LSTM with config."""
        super().__init__()

        self.c = config
        self.z_size = 4
        self.lstm_size = 256
        img_size = self.c.channels * self.c.width * self.c.height

        self.rnn = nn.LSTMCell(img_size, self.lstm_size)
        self.fc1 = nn.Linear(self.lstm_size, 50)
        self.fc2 = nn.Linear(50, 2 * self.z_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        torch.nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, frames):
        """Apply rnn to predict z_where.

        Written as debugging b/c of sub-par cnn performance.
        Args:
            frames (torch.Tensor), (-1, c, w, h): Images.
        Returns:
            zp (torch.Tensor), (-1, O, 8): Object state mean and stds for pos
                and scale.

        """
        # flatten to (n4, cwh)
        x_flat = frames.flatten(start_dim=1)
        batch_size = x_flat.size(0)

        # init vars
        zps = []
        h_enc = torch.zeros(batch_size, self.lstm_size, device=self.c.device, dtype=self.c.dtype)
        c_enc = torch.zeros(batch_size, self.lstm_size, device=self.c.device, dtype=self.c.dtype)

        for _ in range(self.c.num_obj):

            h_enc, c_enc = self.rnn(x_flat, (h_enc, c_enc))
            zp = torch.sigmoid(self.fc1(h_enc))
            # shape is (nT, 8) supair params per object
            zp = self.fc2(zp)
            zps.append(zp)

        # stack along objects to (-1, o, 8)
        zps = torch.stack(zps, 1)
        return zps
