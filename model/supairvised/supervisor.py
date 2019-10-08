import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Exponential, MultivariateNormal, Normal

import sys
sys.path.append("..")
from encoder import RnnStates as Encoder

class Supervisor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c = config
        self.encoder = Encoder(self.c)

    def constrain_zp(self, zp):
        """Constrain z parameter values to sensible ranges
        First 4 are means, last 4 are vars.
        :param zp: (nTo, 8)
        :returns zp_mean, zp_std: shapes 2 x (nTo, 4)
        """

        # constrain scales between 0 and 1, positions between -1 and 1
        zp_mean = torch.cat(
            [torch.sigmoid(zp[:, 0:2]),
             2 * torch.sigmoid(zp[:, 2:4]) - 1], -1)

        zp_std = torch.cat(
            [self.c.scale_var * torch.sigmoid(zp[:, 4:6]),
             self.c.pos_var * torch.sigmoid(zp[:, 6:8])], -1)

        # adjust scaling from crazyK. We need sx, sy to be between 0, and 1
        # and x, y to be inbetween -1 and 1. further restrict:
        # remember, for crazyK xy was between 0 and height/width
        obj_scale_delta = self.c.max_obj_scale - self.c.min_obj_scale
        y_scale_delta = self.c.max_y_scale - self.c.min_y_scale

        set_max = torch.DoubleTensor([[
            obj_scale_delta, y_scale_delta,
            self.c.obj_pos_bound, self.c.obj_pos_bound]]).to(self.c.device)
        zp_mean = zp_mean * set_max

        set_min = torch.DoubleTensor([[
            self.c.min_obj_scale, self.c.min_y_scale, 0.0, 0.0]]).to(self.c.device)
        zp_mean = zp_mean + set_min

        return zp_mean, zp_std

    def v_from_state(self, z_sup, z_sup_std=None):
        """Get full state by combining supair states.

        Supair only gives us positions and scales. Need velocities for full state.
        Take positions from previous Supair prediction to get estimate of velocity
        at current time.
        :param z_sup: shape (n, T, o, 4)
        :param z_sup_std: shape (n, T, o, 4)
        :returns z_sup_full: shape (n, T, o, 6), where T=0 is all zeros
        :returns z_sup_std_full: shape (n, T, o, 6), where T=0 is all zeros
        """

        # get velocities as differences between positions
        v = z_sup[:, 1:, :, 2:] - z_sup[:, :-1, :, 2:]

        # keep scales and positions from T
        z_sup_full = torch.cat([z_sup[:, 1:], v], -1)
        # add zeros to keep time index consistent, v_t = x_t - x_t-1
        zeros = torch.zeros(z_sup_full[:, 0:1].shape).double().to(self.c.device)
        z_sup_full = torch.cat([zeros, z_sup_full], 1)

        return z_sup_full

    def match_objects(self, z_sup, z_sup_std=None):
        """Gredily match objects over sequence.

        """
        if z_sup_std is not None:
            z = torch.cat([z_sup, z_sup_std], -1)
        else:
            z = z_sup

        T = z.shape[1]
        num_obj = self.c.num_obj

        # sequence of matched indices
        # list of idx containing object assignments. initialise with any assignment.
        # for each image in sequence
        z_matched = [z[:, 0]]

        # permutations = list(itertools.permutations(range(0, num_obj)))  # unused!
        for t in range(1, T):

            # only used to get indices, do not want gradients
            curr = z[:, t, :, 2:4]
            curr = curr.unsqueeze(1).repeat(1, num_obj, 1, 1)
            prev = z_matched[t-1][..., 2:4]
            prev = prev.unsqueeze(2).repeat(1, 1, num_obj, 1)
            # weird bug in pytorch where detaching before unsqueeze would mess
            # with dimensions
            curr, prev = curr.detach(), prev.detach()

            # shape is now (n, o1, o2)
            # o1 is repeat of current, o2 is repeat of previous
            # indexing along o1, we will go through the current values
            # we want to keep o1 fixed, at find minimum along o2

            errors = ((prev - curr)**2).sum(-1)

            _, idx = errors.min(-1)

            # for an untrained supair, these indices will often not be unique.
            # this will likely lead to problems

            # inject some faults for testing
            # idx[-1, :] = torch.LongTensor([0, 0, 0])
            # idx[-2, :] = torch.LongTensor([1, 1, 0])
            # idx[-3, :] = torch.LongTensor([1, 0, 1])
            # idx[1, :] = torch.LongTensor([2, 2, 2])

            # only do correction for rows which are affected
            # no neighbouring indices can be the same
            # here is the reason why curently only 3 objects are supported
            faults = torch.prod(idx[:, 1:] != idx[:, :-1], -1)
            faults = faults * (idx[:, 0] != idx[:, 2]).long()
            # at these indexes we have to do greedy matching
            num_faults = (1-faults).sum()

            if num_faults > 0:
                # need to greedily remove faults
                f_errors = errors[faults == 0]
                # sum along current objects
                min_indices = torch.zeros(num_faults, num_obj)
                for obj in range(num_obj):
                    # find first minimum
                    _, f_idx = f_errors.min(-1)
                    # for each seq, 1 column to eliminate
                    # set error values at these indices to large value 
                    # s.t. they wont be min again
                    s_idx = f_idx[:, obj]

                    min_indices[:, obj] = s_idx

                    # flatten indices, select correct sequence, then column
                    # (column now selected before row bc transposed)
                    t_idx = torch.arange(s_idx.shape[0]) * num_obj + s_idx

                    tmp = f_errors.permute(0, 2, 1).flatten(end_dim=1)
                    # for all rows
                    tmp[t_idx, :] = 1e12

                    # reshape back to original shape
                    f_errors = tmp.reshape(f_errors.shape).permute(0, 2, 1)

                # fix faults with new greedily matched
                idx[faults == 0, :] = min_indices.long()

            # now instead of selecting with python
            # I will again reshape and do my own indices
            # select along n, o
            offsets = torch.arange(0, idx.shape[0] * num_obj, num_obj)
            offsets = offsets.unsqueeze(1).repeat(1, num_obj)
            idx_flat = idx + offsets
            idx_flat = idx_flat.flatten()
            z_flat = z[:, t].flatten(end_dim=1)

            match = z_flat[idx_flat].reshape(z[:, t].shape)
            z_matched += [match]

        z_matched = torch.stack(z_matched, 1)

        if z_sup_std is not None:
            z_sup_matched, z_sup_std = torch.chunk(z_matched, 2, dim=-1)
            return z_sup_matched, z_sup_std
        else:
            return z_matched

    def fix_supair(self, z, z_std=None):
        """ fix weird misalignments in supair 

        .. warning : this leaks information from future. in inference model. 
            but who says I cant do that? think of it as filtering. who cares.
        """
        if z_std is not None:
            z = torch.cat([z, z_std], -1)

        # find state which has large difference between previous *and* next state
        # this is the weird one

        # get differences between states
        diffs = torch.abs(z[:, 1:, :, :2] - z[:, :-1, :, :2]).detach()

        # for the first one there is no previous
        zeros = torch.zeros([diffs.shape[0], 1, *diffs.shape[2:]],
                            device=self.c.device, dtype=self.c.dtype)
        prev = torch.cat([zeros, diffs], 1)
        # for the last one there is no following
        after = torch.cat([diffs, zeros], 1)

        # get indices of where both diffs are too large
        eps = 0.095
        idxs = (prev > eps) * (after > eps)

        # make table where z_t = (z_t-1 + z_t+1)/2
        smooth = (z[:, :-2] + z[:, 2:]) / 2
        # we now have at t=0: z_2 + z_0, this should go to t=1
        # add zeros in beginning and at end to center
        zeros = torch.zeros([z.shape[0], 1, *z.shape[2:]],
                            device=self.c.device, dtype=self.c.dtype)
        smooth = torch.cat([zeros, smooth, zeros], 1)

        # apply smoothing at idxs
        # right now, idxs is only over positions. we will just assume identical
        # for all dimensions
        idxs = torch.cat(z.shape[-1]//2 * [idxs], -1)

        z_smooth = z
        z_smooth[idxs] = smooth[idxs]

        if z_std is not None:
            z_smooth, z_std_smooth = torch.chunk(z_smooth, 2, dim=-1)
            return z_smooth, z_std_smooth
        else:
            return z_smooth


    def forward(self, x):
        """Get state from SuPair.
        :params: Input image (n, T, c, w, h)
        :returns z_sup_full: Full states from SuPAIR shape (n, T-1, o, 4).
        """

        # obtain partial states (position and variances) from supair by
        # applying supair to all images in sequence. shape (nT, o, 8)
        # supair does not actually have any time_dependence
        T = x.shape[1]

        z_sup = self.encoder(x.flatten(end_dim=1))
        # shape (nTo, 4) scales and positions, discard std
        z_sup, _ = self.constrain_zp(z_sup.flatten(end_dim=1))

        # reshape z_sup to (n, T, o, 4)
        nto_shape = (-1, T, self.c.num_obj, 4)
        z_sup = z_sup.reshape(nto_shape)

        # ah fuck. need to match *before* getting velocities
        z_sup = self.match_objects(z_sup)
        z_sup = self.fix_supair(z_sup)

        # build full states from supair
        # shape (n, T, o, 6), scales, positions and velocities
        # first full state at T=1 (need 2 imgs)
        # one more t needed to get vin
        z_sup_full = self.v_from_state(z_sup)

        # no need to sample, dont want to do inference.
        # ignore scales, remove first state with zeros
        z_sup_full = z_sup_full[:, 1:, :, 2:]

        # rescale to match new crazyK-scaling
        # first scale to vin size (0, 10)
        z_sup_full = torch.cat([
            (z_sup_full[..., :2] + 1) / 2 * 10,
            z_sup_full[..., 2:] / 2 * 10], -1)

        # then also add new weights (see load_data)
        z_sup_full = torch.cat([
            z_sup_full[..., :2] / 5,
            z_sup_full[..., 2:] * 2], -1)

        return z_sup_full
