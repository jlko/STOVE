"""Contains SuPAIR model."""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Exponential, Normal

from ..spn import probabilistic_models as prob
from . import encoder as encoder


class Supair(nn.Module):
    """SuPAIR: Sum-Product Attend-Infer-Repeat.

    Variant of Supair with fixed number of objects.
    """

    def __init__(self, config):
        """Set up model."""
        super().__init__()
        self.c = config
        # set by calling trainer instance
        self.step_counter = 0

        # export some properties for debugging
        self.prop_dict = {}

        # LSTM-based encoder recognition network for object states
        self.encoder = encoder.RnnStates(self.c)

        # Object and Background SPNs
        if self.c.debug_obj_spn:
            self.obj_spn = prob._get_simple_obj(self.c)
        else:
            self.obj_spn = prob._get_obj_spn(self.c, seed=self.c.random_seed)

        if self.c.debug_bg_model:
            self.bg_spn = prob._get_simple_bg(self.c)
        else:
            self.bg_spn = prob._get_bg_spn(self.c, seed=self.c.random_seed)

    def likelihood(self, x, z_obj):
        """Evaluate likelihood of x under model.

        Args:
            x (torch.Tensor), (n, T, c, w, h) The given sequence of observations.
            z_obj (torch.Tensor), (nTO, 4): Samples from z distribution.

        Returns:
            log_p_xz (torch.Tensor), (nT): Image likelihood.

        """
        # reshape x to merge batch and sequences to pseudo batch since spn
        # will work on image basis shape (n4, c, w, h)
        x_img = x.flatten(end_dim=1)

        # 1. Background Likelihood
        # reshape to (n4, o, 4) and extract marginalisation information
        z_img = z_obj.view(-1, self.c.num_obj, 4)
        marginalise_patch, marginalise_bg, overlap_ratios = self.masks_from_z(z_img)
        # flatten, st. shape is (n4, cwh) for both
        img_flat = x_img.flatten(start_dim=1)
        marg_flat = marginalise_bg.flatten(start_dim=1)
        # get likelihood of background under bg_spn, output from (n4, 1) to (n4)
        bg_loglik = self.bg_spn.forward(img_flat, marg_flat)[:, 0]

        # 2. Patch Likelihoods
        # extract patches (n4o, c, w, h) from transformer
        patches = self.patches_from_z(x_img, z_obj)
        # flatten into (n4o, c w_out h_out)
        patches_flat = patches.flatten(start_dim=1)
        marginalise_flat = marginalise_patch.flatten(start_dim=1)
        # (n4o)
        patches_loglik = self.obj_spn.forward(patches_flat, marginalise_flat)[:, 0]
        # scale patch_likelihoods by size of patch to obtain
        # well calibrated likelihoods
        patches_loglik = patches_loglik * z_obj[:, 0] * z_obj[:, 1]
        # shape n4o to n4,o
        patches_loglik = patches_loglik.view(-1, self.c.num_obj)

        # 3. Add Exponential overlap_penalty
        overlap_prior = Exponential(self.c.overlap_beta)
        overlap_log_liks = overlap_prior.log_prob(overlap_ratios)

        # 4. Assemble final img likelihood E_q(z|x)[log p(x, z)]
        # expectation is approximated by a single sample
        patches_loglik = patches_loglik.sum(1)
        overlap_log_liks = overlap_log_liks.sum(1)
        scores = [bg_loglik, patches_loglik, overlap_log_liks]
        scores = torch.stack(scores, -1)
        # shape (n4)
        log_p_xz = scores.sum(-1)

        if ((self.step_counter % self.c.print_every == 0)
                or (self.step_counter % self.c.plot_every == 0)):
            if self.c.debug:
                self.prop_dict['bg'] = bg_loglik.mean().detach()
                self.prop_dict['patch'] = patches_loglik.mean().detach()
                self.prop_dict['overlap'] = overlap_log_liks.mean().detach()
            if self.c.debug and self.c.debug_extend_plots:
                self.prop_dict['overlap_ratios'] = overlap_ratios.detach()
                self.prop_dict['patches'] = patches.detach()
                self.prop_dict['marginalise_flat'] = marginalise_flat.detach()
                self.prop_dict['patches_loglik'] = patches_loglik.detach()
                self.prop_dict['marginalise_bg'] = marginalise_bg.detach()
                self.prop_dict['bg_loglik'] = bg_loglik.detach()

        return log_p_xz, self.prop_dict

    def constrain_zp(self, zp):
        """Constrain z parameter values to sensible ranges.

        Args:
            zp (torch.Tensor), (nTo, 8): Object state distribution parameters.
                First four are means, last for are stds.

        Returns
            zp_mean, zp_std (torch.Tensor), 2 * (nTo, 4): Constrained parameters
                of object state distributions: (scales, positions)

        """
        # constrain scales between 0 and 1, positions between -1 and 1
        zp_mean = torch.cat(
            [torch.sigmoid(zp[:, 0:2]),
             2 * torch.sigmoid(zp[:, 2:4]) - 1], -1)

        # constrain stds
        zp_std = torch.cat(
            [self.c.scale_var * torch.sigmoid(zp[:, 4:6]),
             self.c.pos_var * torch.sigmoid(zp[:, 6:8])], -1)

        # further constrain scales and oblongess
        obj_scale_delta = self.c.max_obj_scale - self.c.min_obj_scale
        y_scale_delta = self.c.max_y_scale - self.c.min_y_scale

        set_max = torch.tensor(
            [[obj_scale_delta, y_scale_delta,
              self.c.obj_pos_bound, self.c.obj_pos_bound]],
            device=self.c.device, dtype=self.c.dtype)
        zp_mean = zp_mean * set_max

        set_min = torch.tensor(
            [[self.c.min_obj_scale, self.c.min_y_scale, 0.0, 0.0]],
            device=self.c.device, dtype=self.c.dtype)
        zp_mean = zp_mean + set_min

        return zp_mean, zp_std

    @staticmethod
    def sy_from_quotient(z):
        """Get [sx, sy, ...] from [sx, sy/sx, ...]."""
        result = torch.stack([z[..., 0], z[..., 0] * z[..., 1]], -1)
        result = torch.cat([result, z[..., 2:]], -1)
        return result

    @staticmethod
    def quotient_from_sy(z):
        """Get [sx, sy/sx, ...] from [sx, sy, ...]."""
        result = torch.stack([z[..., 0], z[..., 1] / z[..., 0]], -1)
        result = torch.cat([result, z[..., 2:]], -1)
        return result

    def get_z_sup_sample(self, zp_mean, zp_std):
        """Get z sample and log_lik of sample from state code.

        Args:
            zp_mean, zp_std (torch.Tensor), 2 * (nTo, 4): State dist. parameters.

        Returns:
            z_obj (torch.Tensor), (nTo, 4): Sampled states.
            log_q_xz (torch.Tensor), (nTo): Likelihood of samples for ELBO.

        """
        # get z from sampling, each gaussian has dim (4)
        # we need n4o samples per gaussian. dim of sampling is again n4o, 4
        z_dist = Normal(zp_mean, zp_std)

        # rsample can propagate gradients, no explicit reparametrization
        z_tmp = z_dist.rsample().to(self.c.device)
        # Get log lik of sample
        # approximated E_q(z|x) [log q(z|x)] with single sample
        # Sum (in log-domain) the probabilities for the z's [per image]
        # sum (n4o, 4) to n4o
        log_q_xz = z_dist.log_prob(z_tmp).sum(-1)

        # Get sy from sy/sx sy.
        z_obj = self.sy_from_quotient(z_tmp)

        return z_obj, log_q_xz

    @staticmethod
    def expand_z(z):
        """Expand z_where for spatial transformers.

        From [sx, sy, x, y] to [[sx, 0, x], [0, sy, y]].
        Args:
            z (torch.Tensor), (nTo, 4): Object states.

        Returns:
            out (torch.Tensor), (nTo, 2, 3): Transformed states.

        """
        # get batch size dynamically (may change for parallel execution)
        n = z.size(0)
        # add a few zeros to z (bc. we need to select them)
        out = torch.cat((z.new_zeros(n, 1), z), 1)
        expansion_indices = torch.LongTensor([1, 0, 3, 0, 2, 4])
        if z.is_cuda:
            expansion_indices = expansion_indices.cuda()

        out = torch.index_select(out, 1, expansion_indices)
        out = out.view(n, 2, 3)

        return out

    @staticmethod
    def invert_z(z):
        """Invert transformation of z_where for spatial transformers.

        Using this form, expand_z will also work on z_inv, since we need to
        obtain the matrix [[1/sx, 0, -x/sx,], [0, 1/sy, -y/sy]] for inverse.

        Args:
            z (torch.Tensor), (nTo, 4): Object states [sx, sy, x, y].

        Returns:
            z_inv (torch.Tensor), (nTo, 4): Inverses [1/sx, 1/sy, -x/sx, -y/sy].


        """
        i1 = 1. / z[:, 0]
        i2 = 1. / z[:, 1]
        i3 = - z[:, 2] / z[:, 0]
        i4 = - z[:, 3] / z[:, 1]
        z_inv = torch.stack([i1, i2, i3, i4], 1)

        return z_inv

    def patches_from_z(self, x_img, z_obj):
        """From z and image get object patches.

        Grid sample expects input of shape (-1, c, *h*, *w*). But I think, we
        can ignore this here. This effectively switches our x and y as given in
        theta around. There should be no side effects from this. (Also have to
        switch w_out and h_out w.r.t torch documentation.)

        Args:
            x_img (torch.Tensor), (nT, c, w, h): Images.
            z_obj (torch.Tensor), (nTo, 4): Object states.

        Returns:
            out (torch.Tensor), (n4o, c, w_out, h_out): Object patches.

        """
        theta = self.expand_z(z_obj)

        # broadcast x s.t. each image is repeated as many times as there are
        # objects in the scene. I checked that images are repeated next to each
        # other, i.e. same as in z, shape (n4o, c, w, h)
        x_obj = x_img.flatten(start_dim=1).repeat(1, self.c.num_obj)
        x_obj = x_obj.view(-1, *x_img.size()[1:])

        # output shape of grid (n4o, c, patch_width, patch_height)
        w_out, h_out = self.c.patch_width, self.c.patch_height

        # grid shape (n4o, w_out, h_out, 2)
        # grid contains parameters of img used for each sample pixel
        # (also note that this ignores channel, just like it should)
        # we take channel from x_obj!
        grid = F.affine_grid(
            theta, torch.Size((x_obj.size(0), x_img.shape[1], w_out, h_out)))

        out = F.grid_sample(x_obj, grid)
        return out

    def masks_from_z(self, z_img):
        """Given z, get background and patch marginalisation information.

        This has to be done object by object (for all n4 images in batch)
        Note that even if we assume a color channel here, this channel is only
        dragged along and all operations are the same for all channels.

        Args:
            z_img (torch.Tensor), (n4, o, 4): Object states.

        Returns:
            marginalise_patch (torch.Tensor), (n4o, c, w_out, h_out): Mask for
                each object. Contains marginalisation information w.r.t. object
                overlap.

            current_background (torch.Tensor), (n4, c, w, h): Marginalisation
                mask for background. Cut-outs where objects are.
            overlap_ratios (torch.Tensor), (n4, o): Proportion of pixels in shape
                which either overlap with patches of previous objects or are out
                of bounds.

        """
        w_out, h_out = self.c.patch_width, self.c.patch_height
        width, height = self.c.width, self.c.height
        channels = self.c.channels

        const_background = z_img.new_ones(z_img.size(0), channels, width, height)
        marginalise_patch = []

        # Image
        # grid_sample needs a channel dimension, add it, makes everything a lot
        # easier than tiling an empty channel for patch marginalisation
        current_background = z_img.new_zeros(z_img.size(0), channels, width, height)

        # for each object, z_step (n4, 4)
        for z_step in z_img.transpose(0, 1):

            # 1: Get current state of background to assemble marginalisation
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

            # 2: Update current_background for next object, by pasting
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

    def spn_max_activation(self, spn=None):
        """Get max activation reconstructions from a single SPN.

        Each object will look the same, just extracts argmax information
        from SPN, no dependency on any input.

        Args:
            spn (str): ['bg', 'obj'] Choose SPN from which to reconstruct.

        """
        if spn is None:
            spn = self.obj_spn

        params = spn.get_sum_params()

        max_idxs = {vec: np.argmax(p.detach().numpy(), 0)
                    for (vec, p) in params.items()}

        mpe_img = spn.reconstruct(max_idxs, 0, sample=False)

        mpe_img = np.clip(mpe_img, 0., 1.)
        mpe_img = torch.Tensor(mpe_img, device=self.c.device).type(self.c.dtype)

        return mpe_img

    def spn_mpe(self, z, x, spn=None):
        """Get MPE reconstruction of an object patch.

        Args:
            z (torch.Tensor), (nT, o, >=4): Object states. T may be 1.
            x (torch.Tensor), (nT, o, c, w, h) or (n, o, ..): Images.
        Returns:
            recons (torch.Tensor), (n, T, o, c, w_out, h_out): Reconstructions
                of object patches in x, with location given by z.

        """
        if spn == 'bg':
            spn = self.bg_spn
        elif spn is None:
            spn = self.obj_spn

        if x.shape[0] != z.shape[0]:
            raise ValueError('x and z need to have same batch_dim.')

        # first we need to get patches from z
        patches = self.patches_from_z(
            x,
            z.flatten(end_dim=1)
            )

        _, child_acts = spn.compute_activations(
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

    def reconstruct_from_z(
            self, z, x=None, max_activation=True, single_image=True):
        """Reconstruct an image x given states z.

        Make sure images and states are aligned.

        Args:
            z (torch.Tensor), (n, T, o, >=4): Object states.
            x (torch.Tensor), (n, T, o, c, w, h): Images. T dimension may be
                left  out if single_image is True. c corresponds to channels
                modelled in spn.
            max_activation (bool): If true, max activation reconstructions are
                returned. They are independent of any input states or images and
                correspond just to the maximum likelihood input to the SPNS. If
                false, actual reconstructions of image patches are constructed.
            single_image (bool): If max_activation is false, in generative
                scenarios, we cannot provide more than one image to reconstruct
                from.
        Returns:
            reconstructions (torch.Tensor), (n, T, o, c, w, h): Reconstructions,
                or, if single_image was true, conditionally generated frames.

        """
        z = z[..., :4]
        ph, pw = self.c.patch_height, self.c.patch_width
        h, w = self.c.height, self.c.width

        bg_appearance = self.spn_max_activation(spn=self.bg_spn).view(w, h)
        # reshape to nT, o, c, h, w  (or w, h... bug potential)
        reconstructions = torch.Tensor(bg_appearance).unsqueeze(0).repeat(
            z.shape[0]*z.shape[1], 1, 1).unsqueeze(1)

        if max_activation:
            obj_patches = self.spn_max_activation(spn=self.obj_spn).view(pw, ph)
            obj_patches = obj_patches.unsqueeze(0).repeat(self.c.num_obj, 1, 1)
            obj_patches = obj_patches.unsqueeze(0).repeat(
                z.shape[0]*z.shape[1], 1, 1, 1)
            obj_patches = obj_patches.view(
                *obj_patches.shape[:2], self.c.channels, *obj_patches.shape[2:])

        else:
            if x is None:
                raise ValueError(
                    'Need x for reconstructions.')
            if single_image:
                z_in = z[:, 0]
                x_in = x
            else:
                z_in = z.flatten(end_dim=1)
                x_in = x.flatten(end_dim=1)

            obj_patches = self.spn_mpe(
                z_in, x_in, spn=self.obj_spn)

            obj_patches = obj_patches.view(
                z_in.shape[0], self.c.num_obj, self.c.channels, pw, ph)
            if single_image:
                obj_patches = obj_patches.unsqueeze(1).repeat(
                    1, z.shape[1], 1, 1, 1, 1).flatten(end_dim=1)

        # nT, o, 4
        z_img = z.flatten(end_dim=1)

        # loop over objects
        for z_step, obj_patch in zip(z_img.transpose(0, 1),
                                     obj_patches.transpose(0, 1)):
            # transform object patches to image size
            inv_theta = self.expand_z(self.invert_z(z_step))
            grid = F.affine_grid(inv_theta, torch.Size((z_img.size(0), 1, w, h)))
            obj_mask = F.grid_sample(obj_patch, grid)
            reconstructions += obj_mask

        return reconstructions.view([*z.shape[:2], self.c.channels, w, h])

    def forward(self, x):
        """Get supair ELBO of given sequence.

        Args:
            x (torch.Tensor), (n, T, c, w, h): Images.

        Returns:
            average_elbo (torch.Tensor), (1): Elbo purely from SuPAIR.
            prop_dict (dict): Dictionary with performance metrics.

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
        log_q_xz = log_q_xz.view(-1, self.c.num_obj).sum(-1)

        # shape nT
        log_p_xz, _ = self.likelihood(x, z_obj)

        # average log lik as error score
        elbo = log_p_xz - log_q_xz
        average_elbo = torch.mean(elbo)

        if ((self.step_counter % self.c.print_every == 0) or
                (self.step_counter % self.c.plot_every == 0)):
            # shape (n, T, o, 4)
            z = z_obj.view(*x.shape[0:2], self.c.num_obj, 4)
            self.prop_dict['z'] = z.detach()
            if self.c.debug:
                self.prop_dict['log_q'] = log_q_xz.mean().detach()
                self.prop_dict['z_std'] = zp_std.mean(0).detach()

            if self.c.debug and self.c.debug_extend_plots:
                self.prop_dict['elbo'] = elbo.detach()
                self.prop_dict['log_q_xz'] = log_q_xz.detach()

        return average_elbo, self.prop_dict
