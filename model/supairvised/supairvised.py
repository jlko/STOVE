import os
import sys
import warnings

import torch
import pandas
import numpy as np
import ast

from setproctitle import setproctitle

from load_data import VinDataset
from supairvised.model import Net
from supairvised.config import SupVinConfig
from supairvised.train import Trainer
from supairvised.supervisor import Supervisor
from utils import load_args


def main(sh_args=None, restore=None, extras=None):
    config = SupVinConfig()

    # either run from comand line or restore from file
    if restore is not None:
        file = os.path.join(restore, 'config.txt')
        config_update = dict(pandas.read_csv(file).to_numpy())
        checkpoint_path = os.path.join(restore, 'checkpoint')
        config_update.update({'checkpoint_path': checkpoint_path})
    else:
        config_update = sh_args

    if extras is not None:
        config_update.update(extras)

    config = load_args(config, config_update)

    if config.supairvised is False:
        raise ValueError('Use vin.py to load standard model.')

    # set device not based on config but based on what is available now
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = config_setup(config)

    # Net Setup
    net = Net(config)
    net = net.to(config.device)
    net = net.type(config.dtype)

    supervisor = Supervisor(config)
    supervisor = supervisor.to(config.device).type(config.dtype)

    # Trainer
    trainer = Trainer(config, net, supervisor)
    if trainer.logger.exp_dir is not None:
        setproctitle(trainer.logger.exp_dir)

    return trainer

def config_setup(config):

    torch.set_num_threads(config.max_threads)

    if isinstance(config.dtype, str):
        config.dtype = eval(config.dtype)

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
    if config.v_mode == 'from_img':
        config.skip = 3

    # set random seed if None
    if config.random_seed is None:
        config.random_seed = np.random.randint(low=0, high=1000)

    return config

if __name__ == '__main__':
    # call script with  --args batch_size 20 encoder cnn plot_every 1 var argument parsing

    keys = sys.argv[2::2]
    values = sys.argv[3::2]
    kvdict = {k: v for k, v in zip(keys, values)}
    main(sh_args=kvdict)
    trainer.train()