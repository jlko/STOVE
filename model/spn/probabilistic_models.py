"""OBJ and BG models."""
from torch.distributions import Normal

from .rat_torch import SpnArgs, RatSpn
from .region_graph import RegionGraph


def _get_obj_spn(c, seed):
    # Create Region Graph
    patch_size = c.channels * c.patch_width * c.patch_height
    rg = RegionGraph(range(patch_size), seed=seed)
    for _ in range(6):
        rg.random_split(2, 2)
    # Create SPN
    spn_args = SpnArgs()
    spn_args.num_gauss = c.obj_spn_num_gauss
    spn_args.num_sums = c.obj_spn_num_sums

    spn_args.gauss_min_sigma = c.obj_min_var
    spn_args.gauss_max_sigma = c.obj_max_var

    if c.debug_spn_mean:
        spn_args.gauss_mean_of_means = 0.9  # 0.4  # 0.9
        print('debugging obj_spn mean to {}'.format(spn_args.gauss_mean_of_means))

    if c.debug_spn_obj_min_mean is not None:
        print('debugging obj_spn mean to {}'.format(spn_args.gauss_mean_of_means))
        spn_args.gauss_min_mean = c.debug_spn_obj_min_mean

    return RatSpn(1, region_graph=rg, args=spn_args, name='obj-spn')


def _get_bg_spn(c, seed):
    # Create Region Graph
    image_size = c.width * c.height * c.channels
    rg = RegionGraph(range(image_size), seed=seed)
    for _ in range(3):
        rg.random_split(2, 1)

    # Create SPN
    spn_args = SpnArgs()
    spn_args.num_gauss = 6
    spn_args.num_sums = 3
    spn_args.gauss_min_sigma = c.bg_min_var
    spn_args.gauss_max_sigma = c.bg_max_var

    if c.debug_spn_mean:
        spn_args.gauss_mean_of_means = 0.123  #  0.4 # 0.123
        print('debugging bg_spn mean to {}'.format(spn_args.gauss_mean_of_means))

    if c.debug_spn_bg_max_mean is not None:
        print('debugging obj_spn mean to {}'.format(spn_args.gauss_mean_of_means))
        spn_args.gauss_max_mean = c.debug_spn_bg_max_mean

    return RatSpn(1, region_graph=rg, args=spn_args, name='bg-spn')


def _get_simple_bg(c):
    var = c.debug_simple_bg_var

    class SimpleBG:
        def __init__(self, var):
            var = var

        def forward(self, img_flat, marg_flat):
            """ Compute likelihood score for the background.

            Assuming a fixed GMM (and black background)
            :param img_flat" (n4, cwh)
            """
            dist = Normal(0.0, var)
            pixel_lls = dist.log_prob(img_flat)
            # only count those pixels for which marg_flat is 0
            # i.e. they do not get marginalised)
            pixel_lls = pixel_lls * (1 - marg_flat)
            # sum accross pixels and batch
            image_lls = pixel_lls.sum(1)
            # unsqueeze to make output (n4, 1) compatible with spn
            return image_lls.unsqueeze(-1)

    return SimpleBG(var)


def _get_simple_obj(c):
    var = c.debug_simple_obj_var

    class SimpleObj:
        def __init__(self, var):
            var = var

        def forward(self, img_flat, marg_flat):
            """ Compute likelihood score for a simple object.

            Assuming a fixed GMM (and white objs)
            :param img_flat" (n4, cwh)
            """
            dist = Normal(0.8, var)
            pixel_lls = dist.log_prob(img_flat)
            # only count those pixels for which marg_flat is 0
            # i.e. they do not get marginalised)
            pixel_lls = pixel_lls * (1 - marg_flat)
            # sum accross pixels and batch
            image_lls = pixel_lls.sum(1)
            # unsqueeze to make output (n4, 1) compatible with spn
            return image_lls.unsqueeze(-1)

    return SimpleObj(var)
