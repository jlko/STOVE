"""Contains Datasets for Trainer class."""

import numpy as np
import pickle
from scipy.io import loadmat
from torch.utils.data import Dataset


def load(path):
    """Load file depending on type.

    Convenience wrapper.

    Args:
        path (str): Valid path to dataset as created by envs/envs.py.

    Returns:
        data (dict): Dataset for use with Stove.

    """
    file_format = path.split('.')[-1]
    if file_format == 'mat':
        return loadmat(path)
    elif file_format == 'pkl':
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        return data
    else:
        raise ValueError('File format {} not recognized.'.format(file_format))


class StoveDataset(Dataset):
    """Dataset class for Stove.

    Usually called from main.py when assembling trainer instance.
    Handles some data preprocessing such as data scaling.
    Works with all datasets creatable by envs/envs.py.
    Works with and without action-conditioned data.

    In data creation, long continuous sequences are created. From these, we
    subsample shorter sequences on which to train the model.
    """

    def __init__(self, config, transform=None, test=False, data=None):
        """Load data and data info."""
        self.c = config
        self.transform = transform

        if data is None:
            if test:
                data = load(self.c.testdata)
            else:
                data = load(self.c.traindata)

        if 'action' in data.keys():
            self.rl = True
            action_space = data['action_space']
        else:
            self.rl = False
            action_space = None

        self.total_img = data['X'][:self.c.num_episodes]
        # Transpose, as PyTorch images have shape (c, h, w)
        self.total_img = np.transpose(self.total_img, (0, 1, 4, 2, 3))

        if (data['y'].shape[0] < self.c.num_episodes) and not test:
            print('WARNING: Data shape smaller than num_episodes specified.')

        self.total_data = data['y'][:self.c.num_episodes]

        if self.rl:
            self.total_actions = data['action'][:self.c.num_episodes]
            # rescale rewards to [0, 1] to make then work with bce loss
            # (standard loss is mse, but bce is option)
            self.total_rewards = data['reward'][:self.c.num_episodes] + 1
            self.total_dones = data['done'][:self.c.num_episodes]

        # Gather information on dataset. This is accessed by main.py, which then
        # sets data specific entries in the config.
        height = self.total_img.shape[-2]
        width = self.total_img.shape[-1]
        coord_lim = data['coord_lim']
        r = data['r']
        self.data_info = {
            'width': width, 'height': height, 'r': r, 'coord_lim': coord_lim,
            'action_space': action_space, 'action_conditioned': self.rl
            }

        if self.c.supairvised:
            # custom rescaling for sup(er/air)vised, only works for billiard?
            self.total_data *= 10 / coord_lim
            self.total_data[..., :2] /=  5
            self.total_data[..., 2:] *= 2
            x = self.total_img.sum(2)
            x = np.clip(x, 0, 1)
            x = np.expand_dims(x, 2)
            self.total_img = x
        else:
            # scale to match native size of STOVE, i.e. pos in [-1, 1]
            self.total_data *= 1 / coord_lim * 2
            self.total_data[..., :2] -= 1

        # clips can start at any frame, but not too late
        num_eps, num_frames = self.total_img.shape[0:2]
        clips_per_ep = num_frames - ((self.c.num_visible +
                                      self.c.num_rollout) *
                                     self.c.frame_step) + 1

        idx_ep, idx_fr = np.meshgrid(list(range(num_eps)),
                                     list(range(clips_per_ep)))

        self.idxs = np.reshape(np.stack([idx_ep, idx_fr], 2), (-1, 2))

    def __len__(self):
        """Len of iterator."""
        return len(self.idxs)

    def __getitem__(self, idx):
        """Use to access sequences.

        Needed for torch DataLoader.
        """
        step = self.c.frame_step

        i, j = self.idxs[idx, 0], self.idxs[idx, 1]

        end_visible = j + self.c.num_visible * step
        end_rollout = end_visible + self.c.num_rollout * step

        present_images = self.total_img[i, j:end_visible:step]
        future_images = self.total_img[i, end_visible:end_rollout:step]

        present = self.total_data[i, j:end_visible:step]
        future = self.total_data[i, end_visible:end_rollout:step]

        sample = {
            'present_images': present_images,
            'future_images': future_images,
            'present_labels': present,
            'future_labels': future,
            }

        if self.rl:
            present_actions = self.total_actions[i, j:end_visible:step]
            future_actions = self.total_actions[i, end_visible:end_rollout:step]

            present_rewards = self.total_rewards[i, j:end_visible:step]
            future_rewards = self.total_rewards[i, end_visible:end_rollout:step]

            present_dones = self.total_dones[i, j:end_visible:step]
            future_dones = self.total_dones[i, end_visible:end_rollout:step]

            sample.update({
                'present_actions': present_actions,
                'future_actions': future_actions,
                'present_rewards': present_rewards,
                'future_rewards': future_rewards,
                'present_dones': present_dones,
                'future_dones': future_dones,
                })

        return sample
