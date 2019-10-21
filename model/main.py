"""Builds Trainer instance from config via shell arguments and or config files."""

import os
import torch
import pandas
import numpy as np
from setproctitle import setproctitle

from .video_prediction.load_data import StoveDataset
from .utils.utils import load_args, str_to_attr


def main(sh_args=None, restore=None, extras=None):
    """Build trainer instance.

    Intended use: main() is either called from __main__. Then, shell arguments
    are loaded via sh_args. Or, main() is used to restore a model and trainer
    from checkpoint. Then, restore is used to specify the model path. And extras
    may be used to update that model's config.

    Args:
        sh_args (dict): Dict with arguments from shell.
        restore (str): Path to run folder which contains checkpoints and config.
        extras (dict): Dict which modifies the arguments loaded from restore.

    Returns:
        Trainer: Initialised trainer instance.

    """
    # get initialised config
    config = build_config(sh_args, restore, extras)

    # data setup
    train_dataset = StoveDataset(config)
    test_dataset = StoveDataset(config, test=True)
    config = load_args(config, train_dataset.data_info)

    if not config.supairvised:
        from .video_prediction.stove import Stove
        from .video_prediction.train import Trainer
    else:
        from .supairvised.supstove import SupStove as Stove
        from .supairvised.train import SupTrainer as Trainer

    # set up model
    stove = Stove(config)
    stove = stove.to(config.device)
    stove = stove.type(config.dtype)

    # trainer
    trainer = Trainer(config, stove, train_dataset, test_dataset)
    if trainer.logger.exp_dir is not None:
        setproctitle(trainer.logger.exp_dir)

    return trainer


def restore_model(restore, extras=None, load='checkpoint'):
    """Restore a model (and not the full trainer) from checkpoint.

    Args:
        restore (str): Path to run folder of model.
        extras (dict): Modify config from checkpoint with this. 'nolog' is
            enabled by default, but may be overridden.
        load (str): Load checkpoint values for parameters. May set to None to
            disable.

    Returns:
        model (Stove): Initialised model instance.

    """
    # default behaviour is to not crowd experiment directory
    # by activating nolog
    if extras is None:
        extras = {'nolog': True}

    elif extras.get('nolog') is None:
        extras.update({'nolog': True})

    else:
        pass

    # initialise config from config.txt and extras
    config = build_config(restore=restore, extras=extras)
    config.checkpoint_path = os.path.join(restore, load)

    if not config.supairvised:
        from .video_prediction.stove import Stove
    else:
        from .supairvised.supstove import SupStove as Stove

    # initialise model from config
    stove = Stove(config)
    stove = stove.to(config.device)
    stove = stove.type(config.dtype)

    if stove.c.checkpoint_path is not None:
        checkpoint = torch.load(
            stove.c.checkpoint_path, map_location=stove.c.device)

        stove.load_state_dict(checkpoint['model_state_dict'])

    return stove


def build_config(sh_args=None, restore=None, extras=None):
    """Set up a config."""
    config_update = {}

    # either run from comand line or restore from file
    if restore is not None:
        file = os.path.join(restore, 'config.txt')
        config_update.update(dict(pandas.read_csv(file).to_numpy()))
        checkpoint_path = os.path.join(restore, 'checkpoint')
        config_update.update({'checkpoint_path': checkpoint_path})

    if sh_args is not None:
        config_update.update(sh_args)

    if extras is not None:
        config_update.update(extras)

    supairvised = str_to_attr(config_update.get('supairvised', False))

    if not supairvised:
        from .video_prediction.config import StoveConfig
    else:
        from .supairvised.config import SupStoveConfig as StoveConfig

    config = StoveConfig()
    config = load_args(config, config_update)

    torch.set_num_threads(config.max_threads)

    if isinstance(config.dtype, str):
        config.dtype = eval(config.dtype)

    # set device not based on config but based on what is available now
    config.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    if config.dtype == torch.double:
        if config.device == torch.device('cpu'):
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif config.device == torch.device('cuda:0'):
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            raise ValueError
    elif config.dtype == torch.float:
        if config.device == torch.device('cpu'):
            torch.set_default_tensor_type(torch.FloatTensor)
        elif config.device == torch.device('cuda:0'):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            raise ValueError
    else:
        raise ValueError

    if config.supair_only:
        config.skip = 0
    else:
        config.skip = 2

    # set seed if None, important for structure of region graph in random spn
    if config.random_seed is None:
        config.random_seed = np.random.randint(low=0, high=1000)

    return config
