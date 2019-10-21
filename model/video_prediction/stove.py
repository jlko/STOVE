"""Contains main STOVE model class."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .dynamics import Dynamics
from .supair import Supair


class Stove(nn.Module):
    """STOVE: Structured object-aware video prediction model.

    This class combines SuPAIR and the dynamics model.
    """

    def __init__(self, config):
        """Set up model."""
        super().__init__()
        self.c = config

        # set by calling trainer instance
        self.step_counter = 0

        # export some properties for debugging
        self.prop_dict = {}

        self.sup = Supair(config)
        self.dyn = Dynamics(config)

        self.reconstruct_from_z = self.sup.reconstruct_from_z

        # Latent prior for unstructured latent at t=0.
        self.latent_prior = Normal(
            torch.Tensor([0], device=self.c.device),
            torch.Tensor([0.01], device=self.c.device))
        self.z_std_prior = Normal(
            torch.Tensor([0.1], device=self.c.device),
            torch.Tensor([0.01], device=self.c.device))

    def v_from_state(self, z_sup):
        """Get full state by combining supair states.

        Supair only gives us positions and scales. Need velocities for full
        state. Take positions from previous Supair prediction to get estimate
        of velocity at current time.

        Args:
            z_sup (torch.Tensor), (n, T, o, 4): Object state means from SuPAIR.

        Returns:
            z_sup_full (torch.Tensor), (n, T, o, 6): Object state means with
                velocities. All zeros at t=0, b/c no velocity available.

        """
        # get velocities as differences between positions
        v = z_sup[:, 1:, :, 2:] - z_sup[:, :-1, :, 2:]

        # keep scales and positions from T
        z_sup_full = torch.cat([z_sup[:, 1:], v], -1)
        # add zeros to keep time index consistent
        zeros = torch.zeros(
            z_sup_full[:, 0:1].shape,
            device=self.c.device, dtype=self.c.dtype)
        z_sup_full = torch.cat([zeros, z_sup_full], 1)

        return z_sup_full

    def v_std_from_pos(self, z_sup_std):
        """Get std on v from std on positions.

        Args:
            z_sup_std (torch.Tensor), (n, T, o, 4): Object state std from SuPAIR.

        Returns:
            z_sup_std_full (torch.Tensor), (n, T, o, 4): Std with added velocity.

        """
        # Sigma of velocities = sqrt(sigma(x1)**2 + sigma(x2)**2)
        v_std = torch.sqrt(
            z_sup_std[:, 1:, :, 2:]**2 + z_sup_std[:, :-1, :, 2:]**2)
        z_sup_std_full = torch.cat([z_sup_std[:, 1:], v_std], -1)
        zeros = torch.zeros(
            z_sup_std_full[:, 0:1].shape,
            device=self.c.device, dtype=self.c.dtype)
        z_sup_std_full = torch.cat([zeros, z_sup_std_full], 1)

        return z_sup_std_full

    def full_state(self, z_dyn, std_dyn, z_sup, std_sup):
        """Sample full state from dyn and supair predictions at time t.

        Args:
            z_dyn, std_dyn (torch.Tensor), 2 * (n, o, cl//2): Object state means
                and stds from dynamics core. (pos, velo, latent)
            z_sup, std_sup (torch.Tensor), 2 * (n, o, 6): Object state means
                and stds from SuPAIR. (size, pos, velo)
        Returns:
            z_s, mean, std (torch.Tensor), 3 * (n, o, cl//2 + 2): Sample of
                full state, SuPAIR and dynamics information combined, means and
                stds of full state distribution.
            log_q (torch.Tensor), (n, o, cl//2 + 2): Log-likelihood of sampled
                state.

        """
        # Get mean of q(z).

        # for scales
        mean_s = z_sup[..., :2]
        std_s = std_sup[..., :2]

        # for latents
        mean_l = z_dyn[..., 4:]
        std_l = std_dyn[..., 4:]

        # for x and v
        m_sup_xv = z_sup[..., 2:6]
        s_sup_xv = std_sup[..., 2:6]

        m_dyn_xv = z_dyn[..., :4]
        s_dyn_xv = std_dyn[..., :4]

        mean_xv = (s_sup_xv**2 * m_dyn_xv + s_dyn_xv**2 * m_sup_xv)
        mean_xv = mean_xv / (s_dyn_xv**2 + s_sup_xv**2)
        std_xv = s_dyn_xv * s_sup_xv / torch.sqrt(s_dyn_xv**2 + s_sup_xv**2)

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
        """Get generative likelihood of obtained transition.

        The generative dyn part predicts the mean of the distribution over the
        new state z_dyn = 'means'. At inference time, a final state prediction
        z = 'results' is obtained together with information from SuPAIR.
        The generative likelihood of that state is evaluated with distribution
        p(z_t| z_t-1) given by dyn. As in Becker-Ehms, while inference q and p
        of dynamics core share means they do not share stds.

        Args:
            means, results (torch.Tensor), (n, T, o, cl//2): Latent object
                states predicted by generative dynamics core and inferred from
                dynamics model and SuPAIR jointly. States contain
                (pos, velo, latent).

        Returns:
            log_lik (torch.Tensor), (n, T, o, 4): Log-likelihood of results
                under means, i.e. of inferred z under generative model.

        """
        # choose std s.t., if predictions are 'bad', punishment should be high
        dist = Normal(means, self.dyn.transition_lik_std)

        log_lik = dist.log_prob(results)

        return log_lik

    def match_objects(self, z_sup, z_sup_std=None, obj_appearances=None):
        """Match objects over sequence.

        No fixed object oder is enforced in SuPAIR. We match object states
        between timesteps by finding the object order which minimizes the
        distance between states at t and t+1.

        Version 1: Worst case complexity O(T*O). Only works for O=num_obj=3.

        Args:
            z_sup, z_sup_std (torch.Tensor), 2 * (n, T, o, 4): SuPAIR state params.
                Contain id swaps b/c SuPAIR has no notion of time.
            obj_appearances (torch.Tensor), (n, T, o, 3): Appearance information
                may be used to aid matching.
        Returns:
            Permuted version of input arguments.

        """
        # scale to 0, 1 to match appearance information which is already in 0, 1
        z = (z_sup + 1)/2
        m_idx = [2, 3]

        if obj_appearances is not None:
            z = torch.cat([z, obj_appearances], -1)
            if self.c.debug_match_appearance:
                # add color channels to comparison
                m_idx += [4, 5, 6]

        if z_sup_std is not None:
            z = torch.cat([z, z_sup_std], -1)

        T = z.shape[1]
        num_obj = self.c.num_obj

        # sequence of matched zs
        z_matched = [z[:, 0]]

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

            # inject some faults for testing
            # idx[-1, :] = torch.LongTensor([0, 0, 0])
            # idx[-2, :] = torch.LongTensor([1, 1, 0])
            # idx[-3, :] = torch.LongTensor([1, 0, 1])
            # idx[1, :] = torch.LongTensor([2, 2, 2])

            """ For an untrained supair, these indices will often not be unique.
                This will likely lead to problems
                Do correction for rows which are affected. How to find them?
                No neighbouring indices can be the same!
                (Here is the reason why curently only 3 objects are supported.)
            """

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
                    f_errors = tmp.view(f_errors.shape).permute(0, 2, 1)

                # fix faults with new greedily matched
                idx[faults == 0, :] = min_indices.long()

            # select along n, o
            offsets = torch.arange(0, idx.shape[0] * num_obj, num_obj)
            offsets = offsets.unsqueeze(1).repeat(1, num_obj)
            idx_flat = idx + offsets
            idx_flat = idx_flat.flatten()
            z_flat = z[:, t].flatten(end_dim=1)

            match = z_flat[idx_flat].view(z[:, t].shape)
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

    def _match_objects(self, z_sup, z_sup_std=None, obj_appearances=None):
        """Match objects over sequence.

        No fixed object oder is enforced in SuPAIR. We match object states
        between timesteps by finding the object order which minimizes the
        distance between states at t and t+1.

        Version 2: Time complexity O(T). Works for all num_obj. BUT: does not
        ensure, that a valid permutation is obtained, i.e., one object may be
        matched multiple times. This has effects largely in the beginning of
        training.

        Args:
            z_sup, z_sup_std (torch.Tensor), 2 * (n, T, o, 4): SuPAIR state params.
                Contain id swaps b/c SuPAIR has no notion of time.
            obj_appearances (torch.Tensor), (n, T, o, 3): Appearance information
                may be used to aid matching.
        Returns:
            Permuted version of input arguments.

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
        batch_size = z.shape[0]

        # sequence of matched indices
        # list of idx containing object assignments. initialise with any assignment.
        # for each image in sequence
        z_matched = [z[:, 0]]

        for t in range(1, T):

            # only used to get indices, do not want gradients
            curr = z[:, t, :, m_idx]
            prev = z_matched[t-1][..., m_idx]
            # prev changes along rows, curr along columns
            prev = prev.unsqueeze(1).repeat(1, num_obj, 1, 1)
            curr = curr.unsqueeze(2).repeat(1, 1, num_obj, 1)

            # weird bug in pytorch where detaching before unsqueeze would mess
            # with dimensions
            curr, prev = curr.detach(), prev.detach()
            # shape is now (n, o1, o2)
            errors = ((prev - curr)**2).sum(-1)

            # get row-wise minimum, these are column indexes (n, o) for matching
            _, col_idx = errors.min(-2)
            col_idx = col_idx.flatten()
            row_idx = torch.arange(0, col_idx.shape[0])
            # contains for each row (N*num_obj) the col index
            # map to 1d index
            idx = row_idx * num_obj + col_idx

            # from this we obtain permutation matrix by filling zero matrices
            # with ones at min_idxs with regularily increasing rows.
            permutation = torch.zeros(
                (batch_size * num_obj * num_obj),
                dtype=self.c.dtype, device=self.c.device)
            permutation[idx] = 1
            permutation = permutation.view(batch_size, num_obj, num_obj)
            # permute input
            z_perm = torch.matmul(permutation, z[:, t])
            z_matched += [z_perm]

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
        """Fix misalignments in SuPAIR.

        SuPAIR sometimes glitches. We detect these glitches and replace them by
        averaging the previous and following states, i.e. z_t becomes
        z_t = 0.5 * (z_t-1 + z_t+1).

        Args:
            z, z_std (torch.Tensor), 2 * (n, T, o, 6): SuPAIR states.

         fix weird misalignments in supair

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

    def object_embedding(self, z, x_color):
        """Obtain simple object apperance embedding from object patches.

        Here, we just take the channel-wise mean.

        Args:
            z (torch.Tensor), (n, T, o, 4): SuPAIR states.

        Returns:
            embedding (torch.Tensor), (n, T, o, 3): Object apperance embedding.

        """
        # use z to get at color values at x
        z_patch = z[..., :4].detach()
        z_patch = self.sup.sy_from_quotient(z_patch)

        patches = self.sup.patches_from_z(
            x_color.flatten(end_dim=1),
            z_patch.flatten(end_dim=2))

        # could do autoencoding or cnn here
        embedding = patches.mean((-1, -2))

        embedding = embedding.view([*z.shape[:-1], 3])
        return embedding

    def stove_forward(self, x, actions=None, x_color=None):
        """Forward pass of STOVE.

        n (batch_size), T (sequence length), c (number of channels), w (image
        width), h (image_height)

        Args:
            x (torch.Tensor), (n, T, c, w, h): Sequences of images.
            actions (torch.Tensor), (n, T): Actions for action-conditioned video
                prediction.

        Returns:
            average_elbo (torch.Tensor) (1,): Mean ELBO over sequence.
            self.prop_dict (dict): Dictionary containing performance metrics.
                Used for logging and plotting.

        """
        T = x.shape[1]
        skip = self.c.skip
        cl = self.c.cl

        # 1. Obtain SuPAIR states.
        # 1.1 Obtain partial states (sizes and positions).
        # apply supair to all images in sequence. shape (nT, o, 8)
        # supair does not actually have any time_dependence
        z_sup = self.sup.encoder(x.flatten(end_dim=1))
        # shape (nTo, 4) scales and positions
        z_sup, z_sup_std = self.sup.constrain_zp(z_sup.flatten(end_dim=1))
        # reshape z_sup to (n, T, o, 4)
        nto_shape = (-1, T, self.c.num_obj, 4)
        z_sup = z_sup.view(nto_shape)
        z_sup_std = z_sup_std.view(nto_shape)

        # extract appearance information
        if self.c.debug_core_appearance or self.c.debug_match_appearance:
            _obj_appearances = self.object_embedding(z_sup, x_color)
        else:
            _obj_appearances = None

        # 1.2 Find consistent ordering of states.
        # get object embedding from states for latent space
        z_sup, z_sup_std, obj_appearances = self.match_objects(
            z_sup, z_sup_std, _obj_appearances)

        if self.c.debug_core_appearance:
            core_appearances = obj_appearances.transpose(0, 1)
        else:
            core_appearances = T * [None]

        # 1.3 Smooth SuPAIR states.
        if self.c.debug_fix_supair:
            z_sup, z_sup_std = self.fix_supair(z_sup, z_sup_std)

        # 1.4 Build full states from supair.
        # shape (n, T, o, 6), scales, positions and velocities
        # first full state at T=1 (need 2 imgs)
        z_sup_full = self.v_from_state(z_sup)
        z_sup_std_full = self.v_std_from_pos(z_sup_std)

        # 2. Dynamics Loop.
        # 2.1 Initialise States.
        # At t=0 we have no dyn, only partial state from supair. see above.
        # At t=1 we have no dyn, however can get full state from supair via
        # supair from t=0. This is used as init for dyn.
        prior_shape = (*z_sup_full[:, skip-1].shape[:-1], cl//2-4)

        init_z = torch.cat(
            [z_sup_full[:, skip-1],
             self.latent_prior.rsample(prior_shape).squeeze()
             ], -1)

        z = (skip-1) * [0] + [init_z] + (T-skip) * [0]
        log_z = T * [0]

        # Get full time evolution of dyn states. Build step by step.
        # (for sup we already have full time evolution). Initialise z_dyn_std
        # with supair values...
        z_dyn = T * [0]
        dyn_std_init = torch.cat([
            z_sup_std_full[:, skip-1, :, 2:],
            self.z_std_prior.rsample(prior_shape).squeeze()
            ], -1)

        z_dyn_std = (skip-1) * [0] + [dyn_std_init] + (T-skip) * [0]
        z_std = (skip-1) * [0] \
            + [torch.cat([z_sup_std_full[:, skip-1, :, :2], dyn_std_init], -1)] \
            + (T-skip) * [0]

        z_mean = T * [0]
        rewards = []

        if actions is not None:
            core_actions = actions.transpose(0, 1)
        else:
            core_actions = T * [None]

        # 2.2 Loop over sequence and do dynamics prediction.
        for t in range(skip, T):
            # core ignores object sizes
            tmp, reward = self.dyn(
                z[t-1][..., 2:], 0, core_actions[t-1], core_appearances[t-1])
            rewards.append(reward)
            z_dyn_tmp, z_dyn_std_tmp = tmp[..., :cl//2], tmp[..., cl//2:]

            z_dyn_tmp, z_dyn_std[t] = self.dyn.constrain_z_dyn(
                z_dyn_tmp, z_dyn_std_tmp)

            z_dyn[t] = torch.cat(
                [z[t-1][..., 2:4] + z_dyn_tmp[..., :2],
                 z_dyn_tmp[..., 2:]
                 ], -1)

            # obtain full state parameters combining dyn and supair
            z[t], log_z[t], z_mean[t], z_std[t] = self.full_state(
                z_dyn[t], z_dyn_std[t], z_sup_full[:, t], z_sup_std_full[:, t])

        # 2.3 Stack results from t=skip:T.
        # stack states to (n, T-2, o, 6)
        z_s = torch.stack(z[skip:], 1)
        z_dyn_s = torch.stack(z_dyn[skip:], 1)
        log_z_s = torch.stack(log_z[skip:], 1)
        z_dyn_std_s = torch.stack(z_dyn_std[skip:], 1)
        z_std_s = torch.stack(z_std[skip:], 1)

        if self.c.action_conditioned:
            rewards = torch.stack(rewards, 1)
        else:
            rewards = torch.Tensor(rewards)

        # 3. Assemble sequence ELBO.
        # 3.1 p(x|z) via SPNs.
        # flatten to (n(T-2)o, 6) and turn sy/sx to sy for likelihood eval
        z_f = self.sup.sy_from_quotient(z_s.flatten(end_dim=2))
        img_lik, sup_prop = self.sup.likelihood(x[:, skip:], z_f[..., :4])
        self.prop_dict.update(sup_prop)
        # also get lik of initial supair
        z_sup_tmp = self.sup.sy_from_quotient(z_sup[:, 1:skip])
        img_lik_sup, _ = self.sup.likelihood(x[:, 1:skip], z_sup_tmp.flatten(end_dim=2))

        # 3.2. Get q(z|x), sample log-likelihoods of inferred z states (n(T-2)).
        log_z_f = log_z_s.sum((-2, -1)).flatten()

        # 3.3 Get p(z_t|z_t-1), generative dynamics distribution.
        trans_lik = self.transition_lik(means=z_dyn_s, results=z_s[..., 2:])
        # sum and flatten shape (n, T-2, o, 6) to (n(T-2))
        trans_lik = trans_lik.sum((-2, -1)).flatten(end_dim=1)

        # 3.4 Finally assemble ELBO.
        elbo = trans_lik + img_lik - log_z_f
        average_elbo = torch.mean(elbo) + torch.mean(img_lik_sup)

        if ((self.step_counter % self.c.print_every == 0) or
                (self.step_counter % self.c.plot_every == 0)):

            self.prop_dict['z'] = self.sup.sy_from_quotient(
                z_s).detach()
            self.prop_dict['z_dyn'] = z_dyn_s.detach()
            self.prop_dict['z_sup'] = self.sup.sy_from_quotient(
                z_sup_full[:, skip:]).detach()
            self.prop_dict['z_std'] = z_std_s.mean((0, 1, 2)).detach()
            self.prop_dict['z_dyn_std'] = torch.cat(
                [torch.Tensor([float('nan'), float('nan')], device=self.c.device),
                 z_dyn_std_s[..., :4].mean((0, 1, 2)).detach()])
            self.prop_dict['z_sup_std'] = z_sup_std_full[:, skip:].mean(
                (0, 1, 2)).detach()
            self.prop_dict['log_q'] = log_z_f.mean().detach()
            self.prop_dict['translik'] = trans_lik.mean().detach()

            if obj_appearances is not None:
                self.prop_dict['obj_appearances'] = obj_appearances[:, skip:].detach()
            else:
                self.prop_dict['obj_appearances'] = None

            if self.c.debug and self.c.debug_extend_plots:
                self.prop_dict['z_dyn_std_full'] = z_dyn_std_s.detach()

        return average_elbo, self.prop_dict, rewards

    def rollout(self, z_last, num=None, sample=False, return_std=False,
                actions=None, appearance=None):
        """Rollout a given state using the dynamics model.

        Args:
            z_last (torch.Tensor), (n, o, cl//2 + 2): Object states as produced,
                e.g., by prop_dict['z'] in vin_forward(). Careful z_last
                contains [sx, sy] and not [sx, sy/sx].
            num (int): Number of states to roll out.
            sample (bool): Sample from distribution from dynamics core instead
                of predicting the mean.
            return_std (bool): Return std of distribution from dynamics model.
            actions torch.Tensor, (n, T): Actions to apply to dynamics model,
                affecting the rollout.
            appearance torch.Tensor, (n, T, o, 3): Appearance information, as
                aid for dynamics model. Assumed constant during rollout.

        Returns:
            z_pred: (n, num, o, cl//2 + 2) Predictions over future states.

        """
        cl = self.c.cl
        if num is None:
            num = self.c.num_rollout

        z = [z_last]
        # keep scale constant during rollout
        scale = z_last[..., :2]
        rewards = []
        # need last z_dyn_std
        if sample or return_std:
            log_qs = []
        # std for first state not given
        if return_std:
            z_dyn_stds = []
        if actions is not None:
            core_actions = actions.transpose(0, 1)
        else:
            core_actions = num * [None]

        for t in range(1, num+1):
            tmp, reward = self.dyn(
                z[t-1][..., 2:], 0, core_actions[t-1], appearance)
            z_tmp, z_dyn_std_tmp = tmp[..., :cl//2], tmp[..., cl//2:]
            rewards.append(reward)

            z_tmp, z_dyn_std = self.dyn.constrain_z_dyn(z_tmp, z_dyn_std_tmp)

            z_tmp = torch.cat(
                [z[t-1][..., 2:4] + z_tmp[..., :2],
                 z_tmp[..., 2:]
                 ], -1)

            if sample or return_std:
                z_dyn_stds.append(z_dyn_std)
                if sample:
                    dist = Normal(z_tmp, z_dyn_std)
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
            z_dyn_stds = torch.stack(z_dyn_stds, 1)
            return z_full, z_dyn_stds.detach(), rewards
        else:
            return z_full, rewards

    def forward(self, x, step_counter, actions=None, pretrain=False):
        """Forward function.

        Can be used to train (action-conditioned) video prediction or
        SuPAIR only without any dynamics model.

        Args:
            x (torch.Tensor), (n, T, o, 3, w, h): Color images..
            step_counter (int): Current training progress.
            actions (torch.Tensor) (n ,T): Actions from env.
            pretrain (bool): Switch for SuPAIR-only training.

        Returns:
            Whatever the respective forwards return.

        """
        self.step_counter = step_counter
        self.sup.step_counter = step_counter
        self.dyn.step_counter = step_counter

        # save color image for apperance information
        x_color = x
        if self.c.debug_bw:
            x = x.sum(2)
            x = torch.clamp(x, 0, 1)
            x = torch.unsqueeze(x, 2)
        else:
            x = x_color

        if pretrain:
            return self.sup(x)
        else:
            if self.c.debug_core_appearance or self.c.debug_match_appearance:
                return self.stove_forward(x, actions=actions, x_color=x_color)
            else:
                return self.stove_forward(x, actions=actions)
