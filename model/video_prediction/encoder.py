import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RnnStates(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c = config
        img_size = self.c.channels * self.c.width * self.c.height
        self.z_size = 4
        self.lstm_size = 256

        self.rnn = nn.LSTMCell(img_size, self.lstm_size)
        self.fc1 = nn.Linear(self.lstm_size, 50)
        self.fc2 = nn.Linear(50, 2 * self.z_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        torch.nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, frames):
        """Apply rnn to predict z_where.

        Written as debugging b/c of sub-par cnn
        performance.
        :param frames: Groups of six input frames of shape (-1, c, w, h)
        :return zp: shape (-1, o, 8)
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


class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c = config
        cl = config.cl

        # each maxpool halves size
        # 10, 5, 2
        self.convs = nn.ModuleList([
            nn.Conv2d(self.c.channels * 3 + 2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ])

        self.x_coord, self.y_coord = self.construct_coord_dims()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 4)

    def construct_coord_dims(self):
        """Build a meshgrid of x, y coordinates to be used as additional channels."""
        x = np.linspace(-1, 1, self.c.patch_width)
        y = np.linspace(-1, 1, self.c.patch_height)
        xv, yv = np.meshgrid(x, y)
        xv = np.reshape(xv, [1, 1, self.c.patch_height, self.c.patch_width])
        yv = np.reshape(yv, [1, 1, self.c.patch_height, self.c.patch_width])
        x_coord = Variable(torch.from_numpy(xv), dtype=self.c.dtype).to(self.c.device)
        y_coord = Variable(torch.from_numpy(yv), dtype=self.c.dtype).to(self.c.device)
        T = self.c.num_visible
        s = self.c.skip
        x_coord = x_coord.expand(self.c.batch_size * (T-s+1), -1, -1, -1)
        y_coord = y_coord.expand(self.c.batch_size * (T-s+1), -1, -1, -1)
        # repeat along stacked dimension
        self.x_coord = x_coord.repeat(3, 1, 1, 1)
        self.y_coord = y_coord.repeat(3, 1, 1, 1)

        return x_coord, y_coord

    def forward(self, x):

        input = torch.cat([x, self.x_coord, self.y_coord], dim=1)

        output = input
        for conv in self.convs:
            output = conv(output)

        # input depth 5 output depth (number of kernels) 16, since padding and stride are 1, output height and width is const
        # Conv2d(5, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) torch.Size([3840, 16, 10, 10])
        # ReLU(inplace) torch.Size([3840, 16, 10, 10])
        # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) torch.Size([3840, 16, 5, 5])
        # Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) torch.Size([3840, 16, 5, 5])
        # ReLU(inplace) torch.Size([3840, 16, 5, 5])
        # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) torch.Size([3840, 16, 2, 2])
        # Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) torch.Size([3840, 32, 2, 2])
        # ReLU(inplace) torch.Size([3840, 32, 2, 2])
        # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) torch.Size([3840, 32, 1, 1])

        out = output.squeeze()
        out2 = F.relu(self.fc1(out))

        # shape (-1, 2)
        out3 = self.fc2(out2)

        return out3


class CnnStates(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.x_coord, self.y_coord = self.construct_coord_dims()

        # Visual Encoder Modules
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        # 2 frames bunched together plus coordinates
        self.conv1 = nn.Conv2d(self.c.channels * 2 + 2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        # shared linear layer to get pair codes of shape N_obj*cl
        self.fc1 = nn.Linear(32, self.c.num_obj * cl)

        # shared MLP to encode pairs of pair codes as state codes N_obj*cl
        self.fc2 = nn.Linear(cl * 2, cl)
        self.fc3 = nn.Linear(cl, cl)

        self.weight_init()

        def weight_init(self):
            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
            torch.nn.init.xavier_uniform_(self.conv3.weight)
            torch.nn.init.xavier_uniform_(self.conv4.weight)
            torch.nn.init.xavier_uniform_(self.conv5.weight)
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.fc3.weight)

            torch.nn.init.constant_(self.self.conv1.bias, 0.1)
            torch.nn.init.constant_(self.self.conv2.bias, 0.1)
            torch.nn.init.constant_(self.self.conv3.bias, 0.1)
            torch.nn.init.constant_(self.self.conv4.bias, 0.1)
            torch.nn.init.constant_(self.self.conv5.bias, 0.1)
            torch.nn.init.constant_(self.self.fc1.bias, 0.1)
            torch.nn.init.constant_(self.self.fc2.bias, 0.1)
            torch.nn.init.constant_(self.self.fc3.bias, 0.1)

    def construct_coord_dims(self):
        """Build a meshgrid of x, y coordinates to be used as additional channels."""
        x = np.linspace(0, 1, self.c.width)
        y = np.linspace(0, 1, self.c.height)
        xv, yv = np.meshgrid(x, y)
        xv = np.reshape(xv, [1, 1, self.c.height, self.c.width])
        yv = np.reshape(yv, [1, 1, self.c.height, self.c.width])
        x_coord = Variable(torch.from_numpy(xv)).to(self.c.device)
        y_coord = Variable(torch.from_numpy(yv)).to(self.c.device)
        x_coord = x_coord.expand(self.c.batch_size * 5, -1, -1, -1)
        y_coord = y_coord.expand(self.c.batch_size * 5, -1, -1, -1)
        return x_coord, y_coord

    def forward(self, frames):
        """
        Apply visual encoder
        :param frames: Groups of six input frames of shape (n, 6, c, w, h)
        :return: State codes of shape (n, 4, o, cl)
        """
        batch_size = self.c.batch_size
        cl = self.c.cl
        num_obj = self.c.num_obj

        pairs = []
        for i in range(frames.shape[1] - 1):
            # pair consecutive frames (n, 2c, w, h)
            pair = torch.cat((frames[:, i], frames[:, i+1]), 1)
            pairs.append(pair)

        num_pairs = len(pairs)
        pairs = torch.cat(pairs, 0)
        # add coord channels (n * num_pairs, 2c + 2, w, h)
        pairs = torch.cat([pairs, self.x_coord, self.y_coord], dim=1)

        # apply ConvNet to pairs
        ve_h1 = F.relu(self.conv1(pairs))
        ve_h1 = self.pool(ve_h1)
        ve_h2 = F.relu(self.conv2(ve_h1))
        ve_h2 = self.pool(ve_h2)
        ve_h3 = F.relu(self.conv3(ve_h2))
        ve_h3 = self.pool(ve_h3)
        ve_h4 = F.relu(self.conv4(ve_h3))
        ve_h4 = self.pool(ve_h4)
        ve_h5 = F.relu(self.conv5(ve_h4))
        ve_h5 = self.pool(ve_h5)

        # pooled to 1x1, 32 channels: (n * num_pairs, 32)
        encoded_pairs = torch.squeeze(ve_h5)
        # final pair encoding (n * num_pairs, o, cl)
        encoded_pairs = self.fc1(encoded_pairs)
        encoded_pairs = encoded_pairs.view(batch_size * num_pairs, num_obj, cl)
        # chunk pairs encoding num_pairs * [(n, o, cl)]
        encoded_pairs = torch.chunk(encoded_pairs, num_pairs)

        triples = []
        for i in range(num_pairs - 1):
            # pair consecutive pairs to obtain encodings for triples
            triple = torch.cat([encoded_pairs[i], encoded_pairs[i+1]], 2)
            triples.append(triple)

        # the triples together, i.e. (n, num_pairs - 1, o, 2 * cl)
        triples = torch.stack(triples, 1)
        # apply MLP to triples
        shared_h1 = F.relu(self.fc2(triples))
        state_codes = self.fc3(shared_h1)
        return state_codes