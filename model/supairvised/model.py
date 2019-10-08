import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        print("Init supairvised IN")
        self.c = config
        self.pool = nn.MaxPool2d(2, 2)
        self.x_coord, self.y_coord = self.construct_coord_dims()
        cl = config.cl

        # Visual Encoder Modules
        self.conv1 = nn.Conv2d(config.channels * 2 + 2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        # shared linear layer to get pair codes of shape N_obj*cl
        self.fc1 = nn.Linear(32, 3 * cl)

        # shared MLP to encode pairs of pair codes as state codes N_obj*cl
        self.fc2 = nn.Linear(cl * 2, cl)
        self.fc3 = nn.Linear(cl, cl)
        # end of visual encoder

        # Interaction Net Core Modules
        # Self-dynamics MLP
        self.self_cores = nn.ModuleList()
        for i in range(3):
            self.self_cores.append(nn.ModuleList())
            self.self_cores[i].append(nn.Linear(cl, cl))
            self.self_cores[i].append(nn.Linear(cl, cl))

        # Relation MLP
        self.rel_cores = nn.ModuleList()
        for i in range(3):
            self.rel_cores.append(nn.ModuleList())
            self.rel_cores[i].append(nn.Linear(1 + cl * 2, 2 * cl))
            self.rel_cores[i].append(nn.Linear(2 * cl, cl))
            self.rel_cores[i].append(nn.Linear(cl, cl))

        # Attention MLP
        self.att_net = nn.ModuleList()
        for i in range(3):
            self.att_net.append(nn.ModuleList())
            self.att_net[i].append(nn.Linear(1 + cl * 2, 2 * cl))
            self.att_net[i].append(nn.Linear(2 * cl, cl))
            self.att_net[i].append(nn.Linear(cl, 1))

        # Affector MLP
        self.affector = nn.ModuleList()
        for i in range(3):
            self.affector.append(nn.ModuleList())
            self.affector[i].append(nn.Linear(cl, cl))
            self.affector[i].append(nn.Linear(cl, cl))
            self.affector[i].append(nn.Linear(cl, cl))

        # Core output MLP
        self.out = nn.ModuleList()
        for i in range(3):
            self.out.append(nn.ModuleList())
            self.out[i].append(nn.Linear(cl + cl, cl))
            self.out[i].append(nn.Linear(cl, cl))

        # Aggregator MLP for aggregating core predictions
        self.aggregator1 = nn.Linear(cl * 3, cl)
        self.aggregator2 = nn.Linear(cl, cl)

        self.diag_mask = 1 - torch.eye(self.c.num_obj).unsqueeze(2).unsqueeze(0).to(self.c.device).double()
        # decoder mapping state codes to actual states
        self.state_decoder = nn.Linear(cl, 4)
        # encoder for the non-visual case
        self.state_encoder = nn.Linear(4, cl)

        self.log_delta_t = torch.nn.parameter.Parameter(torch.Tensor(3))
        self.log_delta_t.data.fill_(0.)

    def construct_coord_dims(self):
        """
        Build a meshgrid of x, y coordinates to be used as additional channels
        """
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

    def core(self, s, core_idx):
        """
        Applies an interaction network core
        :param s: A state code of shape (n, o, cl)
        :param core_idx: The index of the set of parameters to apply (0, 1, 2)
        :return: Prediction of a future state code (n, o, cl)
        """
        # s = torch.cat([(s[..., :2] + 1) / 2, s[..., 2:]], -1)
        # be careful s appears in the end as well!
        s_in = s  # + 1
        self_sd_h1 = F.relu(self.self_cores[core_idx][0](s_in))
        self_dynamic = self.self_cores[core_idx][1](self_sd_h1) + self_sd_h1

        object_arg1 = s_in.unsqueeze(2).repeat(1, 1, self.c.num_obj, 1)
        object_arg2 = s_in.unsqueeze(1).repeat(1, self.c.num_obj, 1, 1)
        distances = (object_arg1[..., 0] - object_arg2[..., 0])**2 +\
                    (object_arg1[..., 1] - object_arg2[..., 1])**2
        distances = distances.unsqueeze(-1)
        combinations = torch.cat((object_arg1, object_arg2, distances), 3)
        rel_sd_h1 = F.relu(self.rel_cores[core_idx][0](combinations))
        rel_sd_h2 = F.relu(self.rel_cores[core_idx][1](rel_sd_h1))
        rel_factors = self.rel_cores[core_idx][2](rel_sd_h2) + rel_sd_h2

        attention = F.relu(self.att_net[core_idx][0](combinations))
        attention = F.relu(self.att_net[core_idx][1](attention))
        attention = torch.exp(self.att_net[core_idx][2](attention))

        # mask out object interacting with itself
        rel_factors = rel_factors * self.diag_mask * attention
        # relational dynamics per object, (n, o, cl)

        rel_dynamic = torch.sum(rel_factors, 2)

        dynamic_pred = self_dynamic + rel_dynamic

        aff1 = torch.tanh(self.affector[core_idx][0](dynamic_pred))
        aff2 = torch.tanh(self.affector[core_idx][1](aff1)) + aff1
        aff3 = self.affector[core_idx][2](aff2)

        # new_s = s[..., 2:] + aff3[..., 2:]
        # new_pos = s[..., :2] + new_s[..., :2] * torch.exp(self.log_delta_t[core_idx])
        # result = torch.cat((new_pos, new_s), 2)
        # result = s + aff3

        aff_s = torch.cat([aff3, s_in], 2)
        out1 = torch.tanh(self.out[core_idx][0](aff_s))
        result = self.out[core_idx][1](out1) + out1
        # debug change here..
        # result = torch.cat([result[..., :2] + s[..., :2],
        #                     result], -1)

        # maybe this is the problem here? no connection between x and v?
        # and we dont have v error
        result = torch.cat([result[..., :2] + s[..., :2],
                            result[..., 2:]], -1)

        return result

    def frames_to_states(self, frames):
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
        # chunk pairs encoding, each is (n, o, cl)
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

    def core_forward(self, x, num_rollout=8, visual=True):
        """
        Rollout a given sequence of observations using the model
        :param x: The given sequence of observations.
        If visual is True, it should be images of shape (n, 6, c, w, h),
                  otherwise states of shape (n, 4, o, 4).
        :param num_rollout: The number of future states to be predicted
        :param visual: Boolean determining the type of input
        :return: rollout_states: predicted future states (n, roll_len, o, 4)
                 present_states: predicted states at the time of
                                 the given observations, (n, 4, o, 4)
        """
        # for i in range(4):
        #     print(i, x[:, :, :, i].min(), x[:, :, :, i].max())
        # get encoded states
        if visual:
            print('Using cnn state encoder. Lol.')
            state_codes = self.frames_to_states(x)
        else:
            state_codes = self.state_encoder(x)
            state_codes[..., :4] = x

        # the 4 state codes (n, o, cl)
        present_states = state_codes[..., :4]
        cur_state_code = state_codes[:, -1]

        rollouts = []
        for i in range(num_rollout):
            # use cores to predict next state using delta_t = 1, 2, 4
            c1 = self.core(cur_state_code, 0)
            # c2 = self.core(s3, 1)
            # c4 = self.core(s1, 2)
            # all_c = torch.cat([c1, c2, c4], 2)
            # aggregator1 = F.relu(self.aggregator1(all_c))
            state_prediction = c1  # self.aggregator2(aggregator1)
            rollouts.append(state_prediction)
            cur_state_code = state_prediction

        rollouts = torch.stack(rollouts, 1)

        rollout_states = rollouts[..., :4]

        return rollout_states, present_states

    def only_predict(self, x, num_rollout=8, visual=True):
        """
        Rollout a given sequence of observations using the model
        Start rolling out right from first state to avoid spikes seen so far in
        supervised.
        """

        # update latent part of cur_state_codes with all images
        # start = 0

        # apply state_encoder on input already
        start = x.shape[1]-4
        if start+1 == x.shape[1]:
            raise ValueError('Choose valid start value.')

        cur_state_code = self.state_encoder(x[:, start])
        for i in range(start+1, x.shape[1]):
            # use cores to predict next state using delta_t = 1, 2, 4
            c1 = self.core(cur_state_code, 0)
            cur_state_code = torch.cat([x[:, i, :, :4], c1[..., 4:]], -1)  # self.aggregator2(aggregator1)

        rollouts = []
        for i in range(num_rollout):
            # use cores to predict next state using delta_t = 1, 2, 4
            c1 = self.core(cur_state_code, 0)
            # c2 = self.core(s3, 1)
            # c4 = self.core(s1, 2)
            # all_c = torch.cat([c1, c2, c4], 2)
            # aggregator1 = F.relu(self.aggregator1(all_c))
            state_prediction = c1  # self.aggregator2(aggregator1)
            rollouts.append(state_prediction)
            cur_state_code = state_prediction

        rollouts = torch.stack(rollouts, 1)
        rollout_states = rollouts[..., :4]

        return rollout_states, x


    def forward(self, x, num_rollout=8, visual=True, debug_rollout=False):
        if not debug_rollout:
            return self.core_forward(x, num_rollout, visual)
        else:
            return self.only_predict(x, num_rollout, visual)
