"""Train and evaluate STOVE."""
import sys

from model.main import main

if __name__ == '__main__':

    # call script with  --args config_option config_value
    # e.g: pythono main.py --args batch_size 20 encoder cnn plot_every 1
    keys = sys.argv[2::2]
    values = sys.argv[3::2]
    kvdict = dict(zip(keys, values))
    trainer = main(sh_args=kvdict)
    trainer.train()
