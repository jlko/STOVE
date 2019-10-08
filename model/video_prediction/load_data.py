from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import torch
import pickle

# this function with fixed offset of 2 for not visible may not be appropriate anymore
def clips_from_episodes(images, labels, visible_l, rollout_l, step):
    """
    Rearrange episodic observations into shorter clips
    :param images: Episodes of images of shape (n, fr, c, h, w)
    :param labels: Episodes of accompanying data for the images (n, fr, obj, d)
    :param visible_l: Number of frames in each clip
    :param rollout_l: Number of future frames for which labels are returned
    :param step: Stepsize for taking frames from the given episodes
    :return: A number of shorter clips (_, visible_l, c, h, w),
             the corresponding labels (_, visible_l, obj, d),
             and future labels (_, rollout_l, obj, d).
    """
    (num_episodes, num_frames, height, width, channels) = images.shape
    num_obj = labels.shape[-2]

    clips_per_episode = num_frames - (rollout_l + visible_l) * step + 1
    num_clips = num_episodes * clips_per_episode

    clips = np.zeros((num_clips, visible_l, height, width, channels))
    present_labels = np.zeros((num_clips, visible_l - 2, num_obj, 4))
    future_labels = np.zeros((num_clips, rollout_l, num_obj, 4))

    for i in range(num_episodes):
        for j in range(clips_per_episode):
            clip_idx = i * clips_per_episode + j

            end_visible = j + visible_l * step
            end_rollout = end_visible + rollout_l * step

            clips[clip_idx] = images[i, j:end_visible:step]
            # dont offset labels and images here, thats unexpected behaviour
            present_labels[clip_idx] = labels[i, j:end_visible:step]
            future_labels[clip_idx] = labels[i, end_visible:end_rollout:step]

    # shuffle
    perm_idx = np.random.permutation(num_clips)
    return clips[perm_idx], present_labels[perm_idx], future_labels[perm_idx]


def load(path):
    file_format = path.split('.')[-1]
    if file_format == 'mat':
        return loadmat(path)
    elif file_format == 'pkl':
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        return data

class VinDataset(Dataset):

    def __init__(self, config, transform=None, test=False):
        self.config = config
        self.transform = transform

        if test:
            data = load(config.testdata)
        else:
            data = load(config.traindata)

        if 'action' in data.keys():
            self.rl = True
            action_space = data['action_space']
        else:
            self.rl = False
            action_space = None

        self.total_img = data['X'][:config.num_episodes]
        # Transpose, as PyTorch images have shape (c, h, w)
        self.total_img = np.transpose(self.total_img, (0, 1, 4, 2, 3))

        if (data['y'].shape[0] < config.num_episodes) and not test:
            print('WARNING: Data shape smaller than num_episodes specified.')

        self.total_data = data['y'][:config.num_episodes]

        if self.rl:
            self.total_actions = data['action'][:config.num_episodes]
            self.total_rewards = data['reward'][:config.num_episodes] + 1
            self.total_dones = data['done'][:config.num_episodes]

        height = self.total_img.shape[-2]
        width = self.total_img.shape[-1]
        coord_lim = data['coord_lim']
        r = data['r']

        self.data_info = {'width': width, 'height': height, 'r': r,
                          'coord_lim': coord_lim, 'action_space': action_space,
                          'action_conditioned': self.rl}

        if coord_lim != 10:
            self.total_data = self.total_data / coord_lim * 10

        if self.config.supairvised:
            self.total_data[..., :2] /= 5
            self.total_data[..., 2:] *= 2

        # 1000, 100
        num_eps, num_frames = self.total_img.shape[0:2]
        # clips can start at any frame, but not too late
        clips_per_ep = num_frames - ((config.num_visible +
                                     config.num_rollout) *
                                     config.frame_step) + 1

        idx_ep, idx_fr = np.meshgrid(list(range(num_eps)),
                                     list(range(clips_per_ep)))

        self.idxs = np.reshape(np.stack([idx_ep, idx_fr], 2), (-1, 2))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        conf = self.config
        step = conf.frame_step

        i, j = self.idxs[idx, 0], self.idxs[idx, 1]

        end_visible = j + conf.num_visible * step
        end_rollout = end_visible + conf.num_rollout * step

        present_images = self.total_img[i, j:end_visible:step]
        future_images = self.total_img[i, end_visible:end_rollout:step]

        # skip first 2 images of sequence bc they are not being predicted
        # not fatal for crazyK bc he did not try supairvised
        present = self.total_data[i, j:end_visible:step]
        future = self.total_data[i, end_visible:end_rollout:step]

        if self.rl:
            present_actions = self.total_actions[i, j:end_visible:step]
            future_actions = self.total_actions[i, end_visible:end_rollout:step]

            present_rewards = self.total_rewards[i, j:end_visible:step]
            future_rewards = self.total_rewards[i, end_visible:end_rollout:step]

            present_dones = self.total_dones[i, j:end_visible:step]
            future_dones = self.total_dones[i, end_visible:end_rollout:step]

            sample = {'present_images': torch.from_numpy(present_images),
                      'future_images': torch.from_numpy(future_images),
                      'present_labels': torch.from_numpy(present),
                      'future_labels': torch.from_numpy(future),
                      'present_actions': torch.from_numpy(present_actions),
                      'future_actions': torch.from_numpy(future_actions),
                      'present_rewards': torch.from_numpy(present_rewards),
                      'future_rewards': torch.from_numpy(future_rewards),
                      'present_dones': torch.from_numpy(present_dones),
                      'future_dones': torch.from_numpy(future_dones),
                      }
        else:
            sample = {'present_images': torch.from_numpy(present_images),
                      'future_images': torch.from_numpy(future_images),
                      'present_labels': torch.from_numpy(present),
                      'future_labels': torch.from_numpy(future),}
        return sample
