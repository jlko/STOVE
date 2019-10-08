import os
import sys
import warnings
import ast

import torch
from setproctitle import setproctitle
import pandas
import numpy as np

from model.video_prediction.model import Net
from model.video_prediction.train import Trainer

from model.video_prediction.config import VinConfig

from model.video_prediction.load_data import VinDataset
from model.utils.utils import str_to_attr, load_args


def main(sh_args=None, restore=None, extras=None):


    # Make models comparable.
    # Maybe not??..

    # Get Config
    config = VinConfig()

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

    config = load_args(config, config_update)

    # set device not based on config but based on what is available now
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config, train_dataset, test_dataset = config_setup(config)

    # Net Setup
    net = Net(config)
    net = net.to(config.device)
    net = net.type(config.dtype)

    # for n, p in net.named_parameters():
    #     print(n, p.shape, p.is_cuda)

    # Trainer
    trainer = Trainer(config, net, train_dataset, test_dataset)
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


    # Data Setup
    if config.data == 'billiards':
        train_dataset = VinDataset(config)
        test_dataset = VinDataset(config, test=True)
        config = load_args(config, train_dataset.data_info)

    elif config.data == 'gravity':
        train_dataset = VinDataset(config)
        test_dataset = VinDataset(config, test=True)
        config = load_args(config, train_dataset.data_info)

    else:
        raise ValueError

    if config.supair_only:
        config.skip = 0
    if config.v_mode == 'from_img':
        config.skip = 3

    # set random seed if None
    if config.random_seed is None:
        config.random_seed = np.random.randint(low=0, high=1000)

    return config, train_dataset, test_dataset


if __name__ == '__main__':

    # call script with  --args config_option config_value
    #e.g: pythono main.py --args batch_size 20 encoder cnn plot_every 1

    keys = sys.argv[2::2]
    values = sys.argv[3::2]
    kvdict = {k: v for k, v in zip(keys, values)}
    trainer = main(sh_args=kvdict)
    trainer.train()
