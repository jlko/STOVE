"""Implement supervisor class."""

from functools import partial
import torch
import torch.nn as nn

from ..video_prediction.encoder import RnnStates
from ..video_prediction.supair import Supair
from ..video_prediction.stove import Stove

class Supervisor(nn.Module):
    """Supervisor extracts states from images using SuPAIR."""

    def __init__(self, config):
        """Initialise class.

        Pick and choose methods from main STOVE code.
        """
        super().__init__()
        self.c = config
        self.encoder = RnnStates(self.c)

        # Granted, this is ugly, but there is no clean way to do this
        # with clean inheritance or composition (that works with pyTorch
        # loading).
        self.constrain_zp = partial(Supair.constrain_zp, self)
        self.v_from_state = partial(Stove.v_from_state, self)
        self.match_objects = partial(Stove.match_objects, self)
        self.fix_supair = partial(Stove.fix_supair, self)

    def forward(self, x):
        """Get state from SuPair.

        Args:
            x (torch.Tensor, (n, T, c, w, h)): Input images.

        Returns:
            z_sup_full(torch.Tensor,(n, T-1, o, 4)): Full states from SuPAIR.

        """
        # obtain partial states (position and variances) from supair by
        # applying supair to all images in sequence. 
        # yields mean and std on position and scales, shape (nT, o, 8)
        T = x.shape[1]
        z_sup = self.encoder(x.flatten(end_dim=1))

        # shape (nTo, 4) scales and positions, discard std
        z_sup, _ = self.constrain_zp(z_sup.flatten(end_dim=1))

        # reshape z_sup to (n, T, o, 4)
        nto_shape = (-1, T, self.c.num_obj, 4)
        z_sup = z_sup.reshape(nto_shape)

        # match before obtaining velocities
        z_sup = self.match_objects(z_sup)

        # smooth supair states
        z_sup = self.fix_supair(z_sup)

        # build full states from supair
        # shape (n, T, o, 6), scales, positions and velocities
        # first full state at T=1 (need 2 imgs)
        # one more t needed to get vin
        z_sup_full = self.v_from_state(z_sup)

        # no need to sample, dont want to do inference.
        # ignore scales, remove first state with zeros
        z_sup_full = z_sup_full[:, 1:, :, 2:]
        
        # scale to custom supervised scale, true data scaled independently
        # of coord_lim already
        z_sup_full = torch.cat([
            (z_sup_full[..., :2] + 1) / 2 * 10,
            z_sup_full[..., 2:] / 2 * 10], -1)
        # then also add new weights (see load_data)
        z_sup_full = torch.cat([
            z_sup_full[..., :2] / 5,
            z_sup_full[..., 2:] * 2], -1)

        return z_sup_full
