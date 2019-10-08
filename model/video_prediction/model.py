import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Exponential, MultivariateNormal, Normal
from torch.distributions.kl import kl_divergence

from ..spn import probabilistic_models as prob
from . import encoder as encoder

class Net(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.c = config
        device = self.c.device

        # export some properties for debugging
        self.prop_dict = {}

        if self.c.encoder == 'cnn':
            self.encoder = encoder.CnnStates(self.c)
        elif self.c.encoder == 'rnn':
            self.encoder = encoder.RnnStates(self.c)
        else:
            raise ValueError

        print(self.c.random_seed)
        # Object and Background SPNs
        if self.c.debug_obj_spn:
            self.obj_spn = prob._get_simple_obj(self.c)
        else:
            self.obj_spn = prob._get_obj_spn(self.c, seed=self.c.random_seed)

        if self.c.debug_bg_model:
            self.bg_spn = prob._get_simple_bg(self.c)
        else:
            self.bg_spn = prob._get_bg_spn(self.c, seed=self.c.random_seed)

        std = self.c.transition_lik_std
        if len(std) == 4:
            std = std + 12 * [0.01]
        elif len(std) == cl // 2:
            pass
        else:
            raise ValueError('Specify valid transition_lik_std.')
        std = torch.Tensor([[std]], device=self.c.device)
        self.transition_lik_std = std

        # Interaction Net Core Modules
        cl = self.c.cl

        enc_input_size = cl//2
        if self.c.action_conditioned:
            # Action Embedding Layer
            self.n_action_enc = 4
            self.action_embedding_layer = nn.Linear(
                self.c.action_space, self.c.num_obj * self.n_action_enc)
            enc_input_size += self.n_action_enc
            # Reward MLP
            self.reward_head0 = nn.Sequential(
                    nn.Linear(cl, cl),
                    nn.ReLU(),
                    nn.Linear(cl, cl),
                    )

            self.reward_head1 = nn.Sequential(
                    nn.Linear(cl, cl//2),
                    nn.ReLU(),
                    nn.Linear(cl//2, cl//4),
                    nn.ReLU(),
                    nn.Linear(cl//4, 1)
                    )

        if self.c.debug_core_appearance:
            enc_input_size += self.c.debug_appearance_dim

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

        self.diag_mask = 1 - torch.eye(
            self.c.num_obj, dtype=self.c.dtype
            ).unsqueeze(2).unsqueeze(0).to(self.c.device)

        self.state_enc = nn.Linear(enc_input_size, cl)

        if self.c.v_mode == 'from_img':
            self.cnn = encoder.SimpleCNN(self.c)

        self.latent_prior = Normal(
            torch.Tensor([0], device=self.c.device),
            torch.Tensor([0.01], device=self.c.device))
        self.z_std_prior = Normal(
            torch.Tensor([0.1], device=self.c.device),
            torch.Tensor([0.01], device=self.c.device))

        if self.c.debug_xavier:
            print('Using xavier init for interaction.')
            self.weight_init()

        self.nonlinear = F.elu if self.c.debug_nonlinear == 'elu' else F.relu

    def weight_init(self):
        for i in range(3):
            for j in range(2):
                torch.nn.init.xavier_uniform_(self.self_cores[i][j].weight)
                torch.nn.init.constant_(self.self_cores[i][j].bias, 0.1)
                torch.nn.init.xavier_uniform_(self.out[i][j].weight)
                torch.nn.init.constant_(self.out[i][j].bias, 0.1)
            for j in range(3):
                torch.nn.init.xavier_uniform_(self.rel_cores[i][j].weight)
                torch.nn.init.constant_(self.rel_cores[i][j].bias, 0.1)
                torch.nn.init.xavier_uniform_(self.att_net[i][j].weight)
                torch.nn.init.constant_(self.att_net[i][j].bias, 0.1)
                torch.nn.init.xavier_uniform_(self.affector[i][j].weight)
                torch.nn.init.constant_(self.affector[i][j].bias, 0.1)

        torch.nn.init.xavier_uniform_(self.aggregator1.weight)
        torch.nn.init.xavier_uniform_(self.aggregator2.weight)
        torch.nn.init.xavier_uniform_(self.state_enc.weight)

        torch.nn.init.constant_(self.aggregator1.bias, 0.1)
        torch.nn.init.constant_(self.aggregator2.bias, 0.1)

        torch.nn.init.constant_(self.state_enc.bias, 0.1)

        torch.nn.init.xavier_uniform_(self.std_layer1.weight)
        torch.nn.init.xavier_uniform_(self.std_layer2.weight)

        torch.nn.init.constant_(self.std_layer1.bias, 0.1)
        torch.nn.init.constant_(self.std_layer2.bias, 0.1)

    # Supair Part
    def likelihood(self, x, z_obj):
        """ Evaluate likelihood of x under model.

        :param x: The given sequence of observations. Shape (n, T, c, w, h).
        :param z_obj: Samples from z distribution (nTo, 4 ). Will assume fixed
            ordering of objects as given by indices, since this is required for
            marginalisation.
        :returns log_p_xz: for each input image, shape (n4)
        """

        # reshape x to merge batch and sequences to pseudo batch since spn
        # will work on image basis shape (n4, c, w, h)
        x_img = x.flatten(end_dim=1)  # end_dim is inclusive!

        # Background Likelihood
        # extract marginalisation information
        # reshape to (n4, o, 4)
        z_img = z_obj.reshape(-1, self.c.num_obj, 4)
        marginalise_patch, marginalise_bg, overlap_ratios = self.masks_from_z(z_img)
        # flatten, st. shape is (n4, cwh) for both
        img_flat = x_img.flatten(start_dim=1)
        marg_flat = marginalise_bg.flatten(start_dim=1)

        # get likelihood of background under bg_spn
        # get output from single node (n4, 1) to (n4)
        bg_loglik = self.bg_spn.forward(img_flat, marg_flat)[:, 0]

        # Patch Likelihoods
        # extract patches (n4o, c, w, h) from transformer
        patches = self.patches_from_z(x_img, z_obj)

        # flatten into (n4o, c w_out h_out)
        patches_flat = patches.flatten(start_dim=1)
        marginalise_flat = marginalise_patch.flatten(start_dim=1)
        # (n4o)
        patches_loglik = self.obj_spn.forward(patches_flat, marginalise_flat)[:, 0]

        # just like supair, scale patch_likelihoods by size of patch to obtain
        # well calibrated likelihoods (pytorch does not like inplace '*=')
        patches_loglik = patches_loglik * z_obj[:, 0] * z_obj[:, 1]

        # from shape n4o to n4,o
        patches_loglik = patches_loglik.reshape(-1, self.c.num_obj)

        # add overlap_penalty. Gamma dist is unstable for overlap=0 in torch
        # alpha=1 for crazyK anyways. use exponential
        overlap_prior = Exponential(self.c.overlap_beta)
        overlap_log_liks = overlap_prior.log_prob(overlap_ratios)

        # Assemble final img likelihood E_q(z|x)[log p(x, z)]
        # expectation is approximated by a single sample
        patches_loglik = patches_loglik.sum(1)
        overlap_log_liks = overlap_log_liks.sum(1)
        scores = [bg_loglik, patches_loglik, overlap_log_liks]
        scores = torch.stack(scores, -1)

        # shape (n4)
        log_p_xz = scores.sum(-1)

        if (self.step_counter % self.c.print_every == 0) or (self.step_counter % self.c.plot_every == 0):
            # mode_specific prefix
            p = ''
            if self.c.debug:
                self.prop_dict['bg'] = bg_loglik.mean().detach()
                self.prop_dict['patch'] = patches_loglik.mean().detach()
                self.prop_dict['overlap'] = overlap_log_liks.mean().detach()
            if self.c.debug and self.c.debug_extend_plots:
                self.prop_dict[p+'overlap_ratios'] = overlap_ratios.detach()
                self.prop_dict[p+'patches'] = patches.detach()
                self.prop_dict[p+'marginalise_flat'] = marginalise_flat.detach()
                self.prop_dict[p+'patches_loglik'] = patches_loglik.detach()
                self.prop_dict[p+'marginalise_bg'] = marginalise_bg.detach()
                self.prop_dict[p+'bg_loglik'] = bg_loglik.detach()

        return log_p_xz

    def constrain_zp(self, zp):
        """Constrain z parameter values to sensible ranges
        First 4 are means, last 4 are vars.
        :param zp: (nTo, 8)
        :returns zp_mean, zp_std: shapes (nTo, 4)
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

        set_max = torch.Tensor([[
            obj_scale_delta, y_scale_delta,
            self.c.obj_pos_bound, self.c.obj_pos_bound]], device=self.c.device)
        zp_mean = zp_mean * set_max

        set_min = torch.Tensor([[
            self.c.min_obj_scale, self.c.min_y_scale, 0.0, 0.0]], device=self.c.device)
        zp_mean = zp_mean + set_min

        return zp_mean, zp_std

    def constrain_z_vin(self, z, z_std=None):
        """ Similar to constraining z_p from supair.

        However, non scales present in z_vin. Instead, we have velocities.

        :param z: shape (n,o,4)
        :param z_std: shape (n,o,4)
        :returns z_c: shape (n,o,4)
        """

        # now predict position differences
        # velocities between -1 and 1 also and then with vel boound
        # should be similar to position changes
        # same for latents
        z_c = torch.cat(
            [0.1 * (2 * torch.sigmoid(z[..., :4]) - 1),
             0.1 * (2 * torch.sigmoid(z[..., 4:]) - 1)
             ], -1)

        # z_std already has positions at this point
        # then constrain velocities
        # then latents
        if z_std is not None:
            # extra stds for latent dimensions
            z_std = torch.cat(
                    [self.c.pos_var * torch.sigmoid(z_std[..., :2]),
                     0.04 * torch.sigmoid(z_std[..., 2:4]),
                     self.c.debug_latent_q_std * torch.sigmoid(z_std[..., 4:])], -1)

            return z_c, z_std

        else:
            return z_c, None

    def sy_from_quotient(self, z):
        """Get [sx, sy, ...] from [sx, sy/sx, ...]."""
        result = torch.stack([z[..., 0], z[..., 0] * z[..., 1]], -1)
        result = torch.cat([result, z[..., 2:]], -1)
        return result

    def quotient_from_sy(self, z):
        """Get [sx, sy/sx, ...] from [sx, sy, ...]."""
        result = torch.stack([z[..., 0], z[..., 1] / z[..., 0]], -1)
        result = torch.cat([result, z[..., 2:]], -1)
        return result

    def get_z_sup_sample(self, zp_mean, zp_std):
        """Get z sample and log_lik of sample from state code.
        :param zp_mean and zp_std: 2 x (nTo, 4)
        :return z_obj: (nTo, 4)
        :return log_q_xz: (nTo)
        """

        # get z from sampling, each gaussian has dim (4)
        # we need n4o samples per gaussian. dim of sampling is again n4o, 4
        z_dist = Normal(zp_mean, zp_std)

        # rsample can propagate gradients
        z_tmp = z_dist.rsample().to(self.c.device)
        # Get log lik of sample
        # Assemble E_q(z|x) [log q(z|x)], again approximated by a single sample.
        # Sum (in log-domain) the probabilities for the z's [per image]
        # now, no reparam trick is needed.
        # sum (n4o, 4) to n4o
        log_q_xz = z_dist.log_prob(z_tmp).sum(-1)

        # Get sy from sy/sx sy.
        z_obj = self.sy_from_quotient(z_tmp)

        return z_obj, log_q_xz

    def expand_z(self, z):
        """Expand z_where.

        From [sx, sy, x, y] to [[sx, 0, x], [0, sy, y]].
        :param z: shape (nTo, 4)
        :param out: shape (nTo, 2, 3)

        """

        # get pseudo batch size dynamically (may change for parallel execution)
        n = z.size(0)
        # add a few zeros to z (bc. we need to select them)
        out = torch.cat((z.new_zeros(n, 1), z), 1)
        expansion_indices = torch.LongTensor([1, 0, 3, 0, 2, 4])
        if z.is_cuda:
            expansion_indices = expansion_indices.cuda()

        out = torch.index_select(out, 1, expansion_indices)
        out = out.view(n, 2, 3)

        return out

    def invert_z(self, z):
        """ 'Invert' transformation of z_where.

        :param z: shape (N, 4) [sx, sy, x, y] (here N=n4o)
        :return: inverse [1/sx, 1/sy, -x/sx, -y/sy]

        Using this form, expand_z will also work on z_inv, since we need to
        obtain the matrix [[1/sx, 0, -x/sx,], [0, 1/sy, -y/sy]]
        """

        i1 = 1. / z[:, 0]
        i2 = 1. / z[:, 1]
        i3 = - z[:, 2] / z[:, 0]
        i4 = - z[:, 3] / z[:, 1]
        z_inv = torch.stack([i1, i2, i3, i4], 1)

        return z_inv

    def patches_from_z(self, x_img, z_obj):
        """From z_where and image get patches.

        :param x_img: shape (nT, c, w, h)
        :param z_obj: shape (nTo, 4)

        Grid sample expects input of shape (N, c, h, w). This effectively
        switches our x and y as given in theta around. There should be no side
        effects from this. (Also have to switch w_out and h_out w.r.t torch
        documentation.)

        :return: output patches shape (n4o, c, w_out, h_out)
        """

        theta = self.expand_z(z_obj)

        # broadcast x s.t. each image is repeated as many times as there are
        # objects in the scene. I checked that images are repeated next to each
        # other, i.e. same as in z, shape (n4o, c, w, h)
        x_obj = x_img.flatten(start_dim=1).repeat(1, self.c.num_obj).view(-1, *x_img.size()[1:])

        # input output shape of grid (n4o, c, patch_width, patch_height)
        w_out, h_out = self.c.patch_width, self.c.patch_height

        # grid shape (n4o, pw, ph, 2) (contains parameters of img used for each sample pixel)
        # (also note that this ignores channel, just like it should)
        # we take channel from x_obj!
        grid = F.affine_grid(theta, torch.Size((x_obj.size(0), x_img.shape[1], w_out, h_out)))

        out = F.grid_sample(x_obj, grid)
        return out

    def masks_from_z(self, z_img):
        """Given z_where, get background and patch marginalisation information.

        This has to be done object by object (for all n4 images in batch)
        :param z_img: (n4, o, 4)

        Note that even if we assume a color channel here, this channel is only
        dragged along and all operations are the same for all channels.

        :returns marginalise_patch: shape (n4o, c, w_out, h_out) mask for each object
            marginalisation map
        :returns current_background: shape (n4, c, w, h)
        :returns overlap_ratios: shape (n4, o) Proportion of pixels in shape
            which either overlap with patches of previous objects or are out
            of bounds.
        """

        w_out, h_out = self.c.patch_width, self.c.patch_height
        width, height = self.c.width, self.c.height
        channels = self.c.channels

        # Patch
        const_background = z_img.new_ones(z_img.size(0), channels, width, height)
        marginalise_patch = []

        # Image
        # grid_sample needs a channel dimension, add it, makes everything a lot
        # easier than tiling an empty channel for patch marginalisation
        current_background = z_img.new_zeros(z_img.size(0), channels, width, height)

        # for each object z_step (n4, 4)
        for z_step in z_img.transpose(0, 1):

            # First: Get current state of background to assemble marginalisation
            # for active object.
            # theta shape (n4, 2, 3), theta to cut out, inv_theta to paste in
            theta = self.expand_z(z_step)

            # check if there were already objects before, project img region to patch
            grid = F.affine_grid(
                theta,
                torch.Size((current_background.size(0), channels, w_out, h_out)))

            # I want to marginalise over out of bounds predictions. Sadly, we can
            # only pad by 0. Therefore invert current background before and
            # after transformation. Padding by 0 therefore becomes padding by one
            # and all is well.
            invert_bg = 1.0 - current_background

            marginalise_patch.append(1.0 - F.grid_sample(invert_bg, grid))  # , mode='nearest'))

            # Second: Update current_background for next object, by pasting
            # rectangle in imaged-size rectangle and subtracting from current
            # background.
            inv_theta = self.expand_z(self.invert_z(z_step))

            # scale window to position of object in img
            # grid shape (n4, width, height, 2)
            grid = F.affine_grid(inv_theta, torch.Size((z_img.size(0), channels, width, height)))
            obj_mask = F.grid_sample(const_background, grid)  # , mode='nearest')

            # subtract object from current background
            current_background = torch.clamp(current_background + obj_mask, 0, 1)

        # shape (n4, o, c, w_out, h_out)
        marginalise_patch = torch.stack(marginalise_patch, 1)

        # get overlap ratios. flatten along image, mean no of white (=1) pixels
        # for us, no of objects is fixed. so we need ratio just on image basis
        overlap_ratios = marginalise_patch.flatten(start_dim=2).mean(dim=2)

        # shape (n4o, c, w_out, h_out) ready for marginalisation
        marginalise_patch = marginalise_patch.flatten(end_dim=1)

        return marginalise_patch, current_background, overlap_ratios

    def spn_max_activation(self, z=None, spn=None):
        """ wrapper to get mpe of spns """
        if spn == 'bg':
            spn = self.bg_spn
        elif spn is None:
            spn = self.obj_spn

        params = spn.get_sum_params()

        max_idxs = {vec: np.argmax(p.detach().numpy(), 0)
                    for (vec, p) in params.items()}

        mpe_img = spn.reconstruct(max_idxs, 0, sample=False)

        mpe_img = np.clip(mpe_img, 0., 1.)
        mpe_img = torch.Tensor(mpe_img, device=self.c.device).type(self.c.dtype)

        return mpe_img

    def spn_mpe(self, z, x, spn=None):
        """ wrapper to get mpe of spns """
        if spn == 'bg':
            spn = self.bg_spn
        elif spn is None:
            spn = self.obj_spn

        if x.shape[0] != z.shape[0]:
            raise ValueError('x and z need to have same batch_dim.')

        # first we need to get patches from z
        patches = self.patches_from_z(
            x,
            z[:, 0, :, :4].flatten(end_dim=1)
            )

        outputs, child_acts = spn.compute_activations(
            patches.flatten(start_dim=1), get_sum_child_acts=True)

        # then propagate those patches through spn to get activations
        recons = []
        for j in range(x.shape[0] * self.c.num_obj):
            max_idxs = {vec: np.argmax(p[j].detach().numpy(), 0)
                        for (vec, p) in child_acts.items()}
            recon = spn.reconstruct(max_idxs, 0, False)
            recon = torch.Tensor(recon, device=self.c.device).type(self.c.dtype)
            recons.append(recon)

        recons = torch.stack(recons, 0)
        recons = np.clip(recons, 0., 1.)
        recons = recons.view(x.shape[0], self.c.num_obj, -1)
        return recons

    def reconstruct_from_z(self, z, x=None, max_activation=True):
        """ reconstruct image given states from spn mpes using spatial transf

        if you choose max_activation=True, make sure x and z are aligned!
        x: (batch_size, c, res, res) is only used to get reconstruction of object once
            per img in batch and is then repeated z.shape[1] times

        """

        ph, pw = self.c.patch_height, self.c.patch_width
        h, w = self.c.height, self.c.width

        bg_appearance = self.spn_max_activation(spn=self.bg_spn).reshape(w, h)
        # reshape to nT, o, c, h, w  (or w, h... bug potential)
        reconstructions = torch.Tensor(bg_appearance).unsqueeze(0).repeat(
            z.shape[0]*z.shape[1], 1, 1).unsqueeze(1)

        if max_activation:
            obj_appearance = self.spn_max_activation(spn=self.obj_spn).reshape(pw, ph)
            obj_appearance = obj_appearance.unsqueeze(0).repeat(self.c.num_obj, 1, 1)
            obj_patches = obj_appearance.unsqueeze(0).repeat(
                z.shape[0]*z.shape[1], 1, 1, 1)

        else:
            if x is None:
                raise ValueError(
                    'Need to specify x, if we want to reconstruct more than max activation.')
            x = torch.Tensor(x, device=self.c.device).type(self.c.dtype)
            # only reconstruct objects once from spn
            obj_appearance = self.spn_mpe(z, x, spn=self.obj_spn).reshape(x.shape[0], self.c.num_obj, pw, ph)
            obj_patches = obj_appearance.unsqueeze(1).repeat(
                1, z.shape[1], 1, 1, 1).flatten(end_dim=1)

        # nT, o, 4
        z_img = z.flatten(end_dim=1)
        # loop over objects
        for z_step, obj_patch in zip(z_img.transpose(0, 1),
                                     obj_patches.transpose(0, 1)):
            # transform object patches to image size
            inv_theta = self.expand_z(self.invert_z(z_step))
            grid = F.affine_grid(inv_theta, torch.Size((z_img.size(0), 1, w, h)))
            obj_mask = F.grid_sample(obj_patch.unsqueeze(1), grid)
            reconstructions += obj_mask

        return reconstructions.reshape([*z.shape[:2], self.c.channels, w, h])

    def supair_forward(self, x):
        """Get supair ELBO of given sequence.

        :param x: The given sequence of observations. images of shape (n, T, c, w, h),
        """
        # obtain state codes of shape (nT, o, 8)
        state_codes = self.encoder(x.flatten(end_dim=1))

        # Sample from z and get log_q_xz for elbo
        # flatten to (nTo, 8)
        zp = state_codes.flatten(end_dim=1)

        # get parameters for sampling. mean and var for sx, sy/sx, x, y
        zp_mean, zp_std = self.constrain_zp(zp)

        # shapes (nTo, 4), (nTo)
        z_obj, log_q_xz = self.get_z_sup_sample(zp_mean, zp_std)

        # sum from (nTo) to (nT, o) to (nT)
        log_q_xz = log_q_xz.reshape(-1, self.c.num_obj).sum(-1)

        # shape nT
        log_p_xz = self.likelihood(x, z_obj)

        # average log lik as error score
        elbo = log_p_xz - log_q_xz

        average_elbo = torch.mean(elbo)

        if (self.step_counter % self.c.print_every == 0) or (self.step_counter % self.c.plot_every == 0):
            # mode_specific prefix
            # shape (n, T, o, 4)
            z = z_obj.reshape(*x.shape[0:2], self.c.num_obj, 4)
            self.prop_dict['z'] = z.detach()
            if self.c.debug:
                self.prop_dict['log_q'] = log_q_xz.mean().detach()
                self.prop_dict['z_std'] = zp_std.mean(0).detach()

            if self.c.debug and self.c.debug_extend_plots:
                self.prop_dict['elbo'] = elbo.detach()
                self.prop_dict['log_q_xz'] = log_q_xz.detach()

        return average_elbo, self.prop_dict

    def core(self, s, core_idx, actions=None, obj_appearances=None):
        """
        Applies an interaction network core
        :param s: A state code of shape (n, o, cl)
        :param core_idx: The index of the set of parameters to apply (0, 1, 2)
        :return: Prediction of a future state code (n, o, cl)
        """
        if actions is not None:
            action_embedding = self.action_embedding_layer(actions)
            action_embedding = action_embedding.view(
                [action_embedding.shape[0], self.c.num_obj, self.n_action_enc])

        # [TEST_REMOVE_FLAG]:
        # add positive offset to positions s.t. relu has no negative effect
        s = s + 1

        if actions is not None:
            s = torch.cat([s, action_embedding], -1)
        if obj_appearances is not None:
            s = torch.cat([s, obj_appearances], -1)

        # add back positions for distance encoding
        s = torch.cat([s[..., :2], self.state_enc(s)[..., 2:]], -1)

        self_sd_h1 = self.nonlinear(self.self_cores[core_idx][0](s))
        self_dynamic = self.self_cores[core_idx][1](self_sd_h1) + self_sd_h1

        object_arg1 = s.unsqueeze(2).repeat(1, 1, self.c.num_obj, 1)
        object_arg2 = s.unsqueeze(1).repeat(1, self.c.num_obj, 1, 1)
        distances = (object_arg1[..., 0] - object_arg2[..., 0])**2 +\
                    (object_arg1[..., 1] - object_arg2[..., 1])**2
        distances = distances.unsqueeze(-1)

        # shape (n, o, o, 2cl+1)
        combinations = torch.cat((object_arg1, object_arg2, distances), 3)
        rel_sd_h1 = self.nonlinear(self.rel_cores[core_idx][0](combinations))
        rel_sd_h2 = self.nonlinear(self.rel_cores[core_idx][1](rel_sd_h1))
        rel_factors = self.rel_cores[core_idx][2](rel_sd_h2) + rel_sd_h2

        attention = self.nonlinear(self.att_net[core_idx][0](combinations))
        attention = self.nonlinear(self.att_net[core_idx][1](attention))
        attention = torch.exp(self.att_net[core_idx][2](attention))

        # mask out object interacting with itself (n, o, o, cl)
        rel_factors = rel_factors * self.diag_mask * attention

        # relational dynamics per object, (n, o, cl)
        rel_dynamic = torch.sum(rel_factors, 2)

        dynamic_pred = self_dynamic + rel_dynamic

        aff1 = torch.tanh(self.affector[core_idx][0](dynamic_pred))
        aff2 = torch.tanh(self.affector[core_idx][1](aff1)) + aff1
        aff3 = self.affector[core_idx][2](aff2)

        aff_s = torch.cat([aff3, s], 2)
        out1 = torch.tanh(self.out[core_idx][0](aff_s))
        result = self.out[core_idx][1](out1) + out1

        if self.c.action_conditioned:
            # reward prediction
            dynamic_pred_rew = dynamic_pred
            reward_data = self.reward_head0(dynamic_pred_rew)
            # sum to (n, cl)
            reward_data = reward_data.sum(1)
            reward = self.reward_head1(reward_data).view(-1, 1)
            reward = torch.sigmoid(reward)

            return result, reward
        else:
            return result, 0

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
        # add zeros to keep time index consistent
        zeros = torch.zeros(z_sup_full[:, 0:1].shape, device=self.c.device, dtype=self.c.dtype)
        z_sup_full = torch.cat([zeros, z_sup_full], 1)

        return z_sup_full

    def v_std_from_pos(self, z_sup_std):
        # Sigma of velocities = sqrt(sigma(x1)**2 + sigma(x2)**2)
        v_std = torch.sqrt(z_sup_std[:, 1:, :, 2:]**2 + z_sup_std[:, :-1, :, 2:]**2)
        z_sup_std_full = torch.cat([z_sup_std[:, 1:], v_std], -1)
        zeros = torch.zeros(z_sup_std_full[:, 0:1].shape, device=self.c.device, dtype=self.c.dtype)
        z_sup_std_full = torch.cat([zeros, z_sup_std_full], 1)
        return z_sup_std_full

    def full_state(self, z_vin, std_vin, z_sup, std_sup):
        """Sample full state from vin and supair predictions at time t.

        :param z_vin: shape (n, o, 4)
        :param std_vin: shape (n, o, 4)
        :param z_sup: shape (n, o, 6)
        :param std_sup: shape (n, o, 6)

        :returns z_s: full state shape (n, o, 6)
        :returns log_q: log_lik of state (n, o, 6)
        :returns std: std of distribution for debugging (n, o, 6)
        """
        # Get mean of q(z).

        # for scales
        mean_s = z_sup[..., :2]
        std_s = std_sup[..., :2]

        # for latents
        mean_l = z_vin[..., 4:]
        std_l = std_vin[..., 4:]

        # for x and v
        m_sup_xv = z_sup[..., 2:6]
        s_sup_xv = std_sup[..., 2:6]

        m_vin_xv = z_vin[..., :4]
        s_vin_xv = std_vin[..., :4]

        mean_xv = (s_sup_xv**2 * m_vin_xv + s_vin_xv**2 * m_sup_xv) / (s_vin_xv**2 + s_sup_xv**2)
        std_xv = s_vin_xv * s_sup_xv / torch.sqrt(s_vin_xv**2 + s_sup_xv**2)

        if self.c.debug_no_latents:
            # do not sample latents
            mean = torch.cat([mean_s, mean_xv], -1)
            std = torch.cat([std_s, std_xv], -1)
            dist = Normal(mean, std)

            z_s = dist.rsample()
            log_q = dist.log_prob(z_s)
            z_s = torch.cat([z_s, torch.zeros_like(mean_l)], -1)

        else:
            mean = torch.cat([mean_s, mean_xv, mean_l], -1)
            std = torch.cat([std_s, std_xv, std_l], -1)
            dist = Normal(mean, std)

            z_s = dist.rsample()
            log_q = dist.log_prob(z_s)

        return z_s, log_q, mean, std

    def transition_lik(self, means, results):
        """ Get likelihood of obtained transition.

        The generative VIN part predicts the mean of the distribution over the
        new state z_vin = means.
        At inference time, a final state prediction z = result is obtained
        together with Supair. The generative likelihood of that state is
        evaluated with distribution (p_z_t| p_z_t-1) given by VIN.
        As in TUDA paper, while inference q and p vor z_IN share means they
        do not share stds.

        :params means: shape (n, T, o, 4)
        :params result: shape (n, T, o, 4)
        :returns log_lik: shape (n, T, o, 4)
        """

        # choose std s.t., if predictions are good, they should fall within 1std of dist
        dist = Normal(means, self.transition_lik_std)

        log_lik = dist.log_prob(results)

        return log_lik

    def v_from_img(self, x, z_sup, z_sup_std):
        """ Get velocity via CNN at fixed glimpse location. """

        # throw away first 2 states. Use 3 States for estimating velocities.
        # t, t-1, and t-2

        vis = x.shape[1]

        # incorporate last 3 images!
        # shape (n,T-2,3,c,w,h)
        x_stacked = torch.stack([x[:, 2:vis], x[:, 1:vis-1], x[:, 0:vis-2]], 2)

        # for each of the 3 * (n(T-2)) images, need to get patch based on
        # z at n(T-2)
        # ignore first 2 images
        z_glimpse = z_sup[:, 2:]
        # need to replicate z along x's stacking axis
        # (n, T-2, o, 4) to (n, T-2, 3, o, 4)
        z_rep = z_glimpse.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        # transform input for patches from z
        # flatten to (n(T-2)3, c, w, h)
        xp = x_stacked.flatten(end_dim=2)
        # flatten to (n(T-2)3o, 4)
        zp = z_rep.flatten(end_dim=-2)
        # extract glimpses
        # shape (n(T-2)3o,c,w_g,h_g)
        patches = self.patches_from_z(xp, zp)
        # shape (n, T-2, 3, o, c, wg, hg)
        patches = patches.reshape([*z_rep.shape[:4], self.c.channels, self.c.patch_width, self.c.patch_height])
        # shape (n, T-2, o, 3, c, wg, hg)
        patches = patches.permute((0, 1, 3, 2, 4, 5, 6))
        # for cnn we want to collapse the first 3 dimensions (n, T-2, o)
        # i.e. per batch_dim per time per object we want to handle 3 images of size 1, 10, 10
        patches = patches.flatten(end_dim=2)
        # now into CNN
        # flatten channel and stack dimension -> n(T-2)o, 3c, wg, hg
        patches = patches.flatten(start_dim=1, end_dim=2)
        # return (n(T-2)o, 4) predicted_velocities (mean and std)
        encoded = self.cnn(patches)

        # split up into velocity and dimension.
        # reshape to (n, T-2, o, 4)
        v, v_std = torch.chunk(encoded.reshape(z_glimpse.shape), 2, -1)

        # do some quick on the fly constraining...
        v = 0.2 * (2 * torch.sigmoid(v) - 1)
        v_std = 0.04 * torch.sigmoid(v_std)

        # add to initial z
        if self.c.debug_cnn:
            # make velocity prediction more essential by calculating new position
            # from velocity.. maybe this will help
            new_pos = z_sup[:, 1:-1, :, 2:4] + v
            new_pos_std = torch.sqrt(z_sup_std[:, 1:-1, :, 2:4]**2 + v_std**2)

            # scale, position, velocity
            z_sup_full = torch.cat([z_sup[:, 2:, :, :2], new_pos, v], -1)
            z_sup_std_full = torch.cat([z_sup_std[:, 2:, :, :2], new_pos_std, v_std], -1)

        else:
            z_sup_full = torch.cat([z_glimpse, v], -1)
            z_sup_std_full = torch.cat([z_sup_std[:, 2:], v_std], -1)

        # add back initial zeros for initial skip of 2 images
        zeros = torch.zeros(z_sup_full[:, :2].shape,
                            device=self.c.device, dtype=self.c.dtype)
        z_sup_full = torch.cat([zeros, z_sup_full], 1)
        zeros = torch.zeros(z_sup_std_full[:, :2].shape,
                            device=self.c.device, dtype=self.c.dtype)
        z_sup_std_full = torch.cat([zeros, z_sup_std_full], 1)

        return z_sup_full, z_sup_std_full

    def _match_objects(self, z_sup, z_sup_std=None):
        """Gredily match objects over sequence.

        .. deprecation_warning : Deprecated in favor of self.match_objects().
            Can match more than 3 objects, but is 20 times slower than new function.
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
        permutations = list(itertools.permutations(range(0, num_obj)))
        for t in range(1, T):
            pos_prev = z_matched[t-1][..., 2:4]

            errors = []
            # find optimal permutation of object ordering compared to ordering at previous timestamp
            for perm in permutations:
                # error is shape n
                errors += [torch.sqrt(((pos_prev - z[:, t, perm, 2:4])**2).sum(-1)).sum(-1)]
                # differences of this perm to previous setup

            # shape (n, T)
            errors = torch.stack(errors, 1)
            # shape n
            _, idx = errors.min(1)

            selector = list(zip(range(idx.shape[0]), idx.cpu().tolist()))
            z_t_matched = [z[i, t, permutations[j]] for i, j in selector]
            z_t_matched = torch.stack(z_t_matched, 0)
            z_matched += [z_t_matched]

        z_matched = torch.stack(z_matched, 1)

        if z_sup_std is not None:
            z_sup_matched, z_sup_std = torch.chunk(z_matched, 2, dim=-1)
            return z_sup_matched, z_sup_std
        else:
            return z_matched

    def __match_objects(self, z_sup, z_sup_std=None):
        """Gredily match objects over sequence.

        """
        if z_sup_std is not None:
            z = torch.cat([z_sup, z_sup_std], -1)
        else:
            z = z_sup

        if self.c.debug_match_v_too:
            end = 6
        else:
            end = 4

        T = z.shape[1]
        num_obj = self.c.num_obj

        # sequence of matched indices
        # list of idx containing object assignments. initialise with any assignment.
        # for each image in sequence
        z_matched = [z[:, 0]]

        # permutations = list(itertools.permutations(range(0, num_obj)))  # unused!
        for t in range(1, T):

            # only used to get indices, do not want gradients
            curr = z[:, t, :, 2:end]
            curr = curr.unsqueeze(1).repeat(1, num_obj, 1, 1)
            prev = z_matched[t-1][..., 2:end]
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

    def match_objects(self, z_sup, z_sup_std=None, obj_appearances=None):
        """Gredily match objects over sequence.

        """

        # scale to 0, 1
        z = (z_sup + 1)/2
        m_idx = [2, 3]

        if obj_appearances is not None:
            # colors are already in 0, 1
            z = torch.cat([z, obj_appearances], -1)

            if self.c.debug_match_appearance:
                # add color channels to comparison
                m_idx += [4, 5, 6]

        if z_sup_std is not None:
            z = torch.cat([z, z_sup_std], -1)

        T = z.shape[1]
        num_obj = self.c.num_obj

        # sequence of matched indices
        # list of idx containing object assignments. initialise with any assignment.
        # for each image in sequence
        z_matched = [z[:, 0]]

        # permutations = list(itertools.permutations(range(0, num_obj)))  # unused!
        for t in range(1, T):

            # only used to get indices, do not want gradients
            curr = z[:, t, :, m_idx]
            curr = curr.unsqueeze(1).repeat(1, num_obj, 1, 1)
            prev = z_matched[t-1][..., m_idx]
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

        # transform back again
        z_sup_matched = 2 * z_matched[..., :4] - 1

        if obj_appearances is not None:
            obj_appearances_matched = z_matched[..., 4:7]

        if z_sup_std is None and obj_appearances is not None:
            return z_sup_matched, obj_appearances_matched

        elif z_sup_std is not None and obj_appearances is None:
            z_sup_std_matched = z_matched[..., 4:8]
            return z_sup_matched, z_sup_std_matched, None

        elif z_sup_std is not None and obj_appearances is not None:
            z_sup_std_matched = z_matched[..., 4+3:8+3]
            return z_sup_matched, z_sup_std_matched, obj_appearances_matched
        else:
            return z_sup_matched


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

    def object_embedding(self, z):
        """ first version, dont use spatial transformers or anything, just use
            color value in object center """
        # use z to get at color values at x
        x = self.color_x
        z_patch = z[..., :4].detach()
        z_patch = self.sy_from_quotient(z_patch)

        patches = self.patches_from_z(
            x.flatten(end_dim=1),
            z_patch.flatten(end_dim=2))

        # could do autoencoding or cnn here
        embedding = patches.mean((-1, -2))

        embedding = embedding.reshape([*z.shape[:-1], 3])
        return embedding


    def vin_forward(self, x, actions=None):
        """Implement forward pass via hmm with IN state transition.
        :param x: shape (n, num_visible, c, w, h). num_visible is number of
            timesteps in the given sequence. I will call it T from now on.

        :returns average_elbo: float
        :returns self.prop_dict: Dictionary containing performance metrics.
            Used for logging and plotting.
        """

        # obtain partial states (position and variances) from supair by
        # applying supair to all images in sequence. shape (nT, o, 8)
        # supair does not actually have any time_dependence
        T = x.shape[1]
        skip = self.c.skip
        cl = self.c.cl

        z_sup = self.encoder(x.flatten(end_dim=1))
        # shape (nTo, 4) scales and positions
        z_sup, z_sup_std = self.constrain_zp(z_sup.flatten(end_dim=1))

        # reshape z_sup to (n, T, o, 4)
        nto_shape = (-1, T, self.c.num_obj, 4)
        z_sup = z_sup.reshape(nto_shape)
        z_sup_std = z_sup_std.reshape(nto_shape)

        if self.c.debug_core_appearance or self.c.debug_match_appearance:
            _obj_appearances = self.object_embedding(z_sup)
        else:
            _obj_appearances = None

        if self.c.debug_greedy_matching:
            if self.c.num_obj != 3:
                raise NotImplementedError('Not Implemented for Quick Matching')

            # get object embedding from states for latent space
            z_sup, z_sup_std, obj_appearances = self.match_objects(
                z_sup, z_sup_std, _obj_appearances)

        else:
            raise NotImplementedError('Not been used for a long time.')

        if self.c.debug_core_appearance:
            core_appearances = obj_appearances
        else:
            core_appearances = None

        if self.c.debug_fix_supair:
            z_sup, z_sup_std = self.fix_supair(z_sup, z_sup_std)

        # build full states from supair
        if self.c.v_mode == 'from_state':
            # shape (n, T, o, 6), scales, positions and velocities
            # first full state at T=1 (need 2 imgs)
            # one more t needed to get vin
            z_sup_full = self.v_from_state(z_sup)
            z_sup_std_full = self.v_std_from_pos(z_sup_std)

        elif self.c.v_mode == 'from_img':
            # first full state at T=2 (need 3 imgs)
            # one more t needed to get vin
            z_sup_full, z_sup_std_full = self.v_from_img(x, z_sup, z_sup_std)

        else:
            raise ValueError

        # At t=0 we have no vin, only partial state from supair. see above.
        # At t=1 we have no vin, however can get full state from supair via
        # supair from t=0. This is used as init for vin.
        prior_shape = (*z_sup_full[:, skip-1].shape[:-1], cl//2-4)

        init_z = torch.cat(
            [z_sup_full[:, skip-1],
             self.latent_prior.rsample(prior_shape).squeeze()
             ], -1)

        z = (skip-1) * [0] + [init_z] + (T-skip) * [0]
        log_z = T * [0]

        # Get full time evolution of vins states. Build step by step.
        # (for sup we already have full time evolution). Initialise z_vin_std
        # with supair values...
        z_vin = T * [0]
        vin_std_init = torch.cat([
            z_sup_std_full[:, skip-1, :, 2:],
            self.z_std_prior.rsample(prior_shape).squeeze()
            ], -1)

        z_vin_std = (skip-1) * [0] + [vin_std_init] + (T-skip) * [0]
        z_std = (skip-1) * [0] \
            + [torch.cat([z_sup_std_full[:, skip-1, :, :2], vin_std_init], -1)] \
            + (T-skip) * [0]

        z_mean = T * [0]

        rewards = []
        for t in range(skip, T):

            # core predicts velocities and latent states. ignore scales.
            if self.c.debug_core_appearance:
                core_appearance = core_appearances[:, t-1]
            else:
                core_appearance = None
            if actions is not None:
                core_action = actions[:, t-1]
            else:
                core_action = None

            tmp, reward = self.core(
                z[t-1][..., 2:], 0, core_action, core_appearance)

            rewards.append(reward)
            z_vin_tmp, z_vin_std_tmp = tmp[..., :cl//2], tmp[..., cl//2:]

            # get new positions from velocities and old positions to assemble full vin_state
            z_vin_tmp, z_vin_std[t] = self.constrain_z_vin(z_vin_tmp, z_vin_std_tmp)

            z_vin[t] = torch.cat(
                [z[t-1][..., 2:4] + z_vin_tmp[..., :2],
                 z_vin_tmp[..., 2:]
                 ], -1)

            # obtain full state parameters combining vin and supair
            z[t], log_z[t], z_mean[t], z_std[t] = self.full_state(
                z_vin[t], z_vin_std[t], z_sup_full[:, t], z_sup_std_full[:, t])

        # Assemble ELBO for predictions from t=2:T
        # stack states to (n, T-2, o, 6)
        z_s = torch.stack(z[skip:], 1)
        z_vin_s = torch.stack(z_vin[skip:], 1)
        log_z_s = torch.stack(log_z[skip:], 1)
        z_vin_std_s = torch.stack(z_vin_std[skip:], 1)
        z_std_s = torch.stack(z_std[skip:], 1)
        z_mean_s = torch.stack(z_mean[skip:], 1)
        if self.c.action_conditioned:
            rewards = torch.stack(rewards, 1)
        else:
            rewards = torch.Tensor(rewards)

        # p(x|z) via spn
        # flatten to (n(T-2)o, 6) and turn sy/sx to sy for likelihood eval
        z_f = self.sy_from_quotient(z_s.flatten(end_dim=2))
        img_lik = self.likelihood(x[:, skip:], z_f[..., :4])
        # also get lik of initial supair
        z_sup_tmp = self.sy_from_quotient(z_sup[:, 1:skip])
        img_lik_sup = self.likelihood(x[:, 1:skip], z_sup_tmp.flatten(end_dim=2))

        # 0. get q(z|x) (not needed for analytic kl, but needed for logging)
        # (n(T-2))
        log_z_f = log_z_s.sum((-2, -1)).flatten()

        # 1. get p(z_t|z_t-1).  vin predicts mean of distributions
        trans_lik = self.transition_lik(means=z_vin_s, results=z_s[..., 2:])

        # sum and flatten shape (n, T-2, o, 6) to (n(T-2))
        trans_lik = trans_lik.sum((-2, -1)).flatten(end_dim=1)

        # 2. get p(x|z) - q(z|x)
        elbo = trans_lik + img_lik - log_z_f

        average_elbo = torch.mean(elbo) + torch.mean(img_lik_sup)

        if (self.step_counter % self.c.print_every == 0) or (self.step_counter % self.c.plot_every == 0):
            self.prop_dict['z'] = self.sy_from_quotient(z_s).detach()
            self.prop_dict['z_vin'] = z_vin_s.detach()
            self.prop_dict['z_sup'] = self.sy_from_quotient(z_sup_full[:, skip:]).detach()
            self.prop_dict['z_std'] = z_std_s.mean((0, 1, 2)).detach()
            self.prop_dict['z_vin_std'] = torch.cat(
                [torch.Tensor([float('nan'), float('nan')], device=self.c.device),
                 z_vin_std_s[..., :4].mean((0, 1, 2)).detach()])
            self.prop_dict['z_sup_std'] = z_sup_std_full[:, skip:].mean((0, 1, 2)).detach()
            if obj_appearances is not None:
                self.prop_dict['obj_appearances'] = obj_appearances[:, skip:].detach()
            else:
                self.prop_dict['obj_appearances'] = None

            if self.c.debug and not self.c.debug_analytic_kl:
                self.prop_dict['log_q'] = log_z_f.mean().detach()
                self.prop_dict['translik'] = trans_lik.mean().detach()

            if self.c.debug and self.c.debug_analytic_kl:
                if (not self.c.debug_deterministic_latent) and (not self.c.debug_no_latents):
                    self.prop_dict['kl_latent'] = kl_latent.mean().detach()
                self.prop_dict['kl_state'] = kl_state.mean().detach()

            if self.c.debug and self.c.debug_extend_plots:
                # self.prop_dict['elbo'] = elbo.detach()
                self.prop_dict['z_vin_std_full'] = z_vin_std_s.detach()
                # self.prop_dict['log_q_xz_o'] = log_z_f.mean().detach()

        return average_elbo, self.prop_dict, rewards

    def rollout(self, z_last, num=None, sample=False, std_init=False, return_std=False,
                actions=None, appearance=None):
        """Test rollout capabilities of physics engine.

        Given an input state, perform rollouts over future states.
        Careful z_last contains sx sy and not sx sy/sx.

        :param z_last: shape (n, o, 6), assume to contain sx and sy
        (vin not trained for this, just pass forward)
        :returns z_pred: shape (n, num_rollout, o, 6) predictions over future
            states (transform to contain sx, sy)
        """
        cl = self.c.cl
        if num is None:
            num = self.c.num_rollout

        z = [z_last]
        # keep scale constant during rollout
        scale = z_last[..., :2]
        rewards = []
        if sample or return_std:
            # need last z_vin_std
            z_vin_stds = [std_init]
            log_qs = []

        for t in range(1, num+1):

            if actions is not None:
                core_action = actions[:, t-1]
            else:
                core_action = None

            tmp, reward = self.core(z[t-1][..., 2:], 0, core_action, appearance)
            z_tmp, z_vin_std_tmp = tmp[..., :cl//2], tmp[..., cl//2:]
            rewards.append(reward)

            z_tmp, z_vin_std = self.constrain_z_vin(z_tmp, z_vin_std_tmp)

            # get new pos from old pos + vel
            z_tmp = torch.cat(
                [z[t-1][..., 2:4] + z_tmp[..., :2],
                 z_tmp[..., 2:]
                 ], -1)

            if sample or return_std:
                z_vin_stds.append(z_vin_std)
                if sample:
                    dist = Normal(z_tmp, z_vin_std)
                    z_tmp = dist.rsample()
                    log_qs.append(dist.log_prob(z_tmp))

            # add back scale
            z_tmp = torch.cat([scale, z_tmp], -1)

            z.append(z_tmp)

        if self.c.action_conditioned:
            rewards = torch.stack(rewards, 1)
        else:
            rewards = torch.Tensor(rewards)

        # first state was given, not part of rollout
        z_full = torch.stack(z[1:], 1)
        if sample:
            return z_full, torch.stack(log_qs, 1), rewards
        if return_std:
            z_vin_stds = torch.stack(z_vin_stds[1:], 1)
            return z_full, z_vin_stds.detach(), rewards
        else:
            return z_full, rewards

    # Forwards
    def forward(self, x, step_counter, actions=None, pretrain=False):
        """x has shape n,T,o,c,res,res"""
        self.step_counter = step_counter
        self.pretrain = pretrain

        self.color_x = x
        if self.c.debug_color or self.c.debug_bw:
            # moved from load_data, s.t. color info is still available
            x = x.sum(2)
            x = torch.clamp(x, 0, 1)
            x = torch.unsqueeze(x, 2)

        if pretrain:
            return self.supair_forward(x)
        else:
            return self.vin_forward(x, actions=actions)
