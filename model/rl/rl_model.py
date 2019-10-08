import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Policy(nn.Module):
    """
    Policy network to be used for reinforcement learning
    """
    def __init__(self, obs_shape, action_space, hidden_size=512, base=None, head=None, stacked_frames=4, use_grid=False, use_deep_layers=False):
        super(Policy, self).__init__()
        self.stacked_frames = stacked_frames
        self.use_grid = use_grid

        if base is None:
            base = SimpleCNNBase
            if self.use_grid:
                in_shape = obs_shape[2] + 2
            else:
                in_shape = obs_shape[2] 
        elif base is 'control':
            base = SimpleControlBase
            in_shape = obs_shape
        else:
            raise NotImplementedError

        if head is None:
            head = DiscreteActionHead
        else:
            raise NotImplementedError

        # TODO: here check for different action spaces which might be applicable
        # could also just fix action_space as action_space.n
        # also need to add a action_space variable/method to the env

        if base.__name__ == 'SimpleCNNBase':
            self.base = base(in_shape, use_grid=self.use_grid)
        else:
            self.base = base(in_shape)
    
        self.head = head(self.base.output_size, action_space, hidden_size=hidden_size, use_deep_layers=use_deep_layers)

    def act(self, inputs, deterministic=False):
        """
        Given an input returns the predicted value and action
        """
        hidden_features = self.base(inputs)
        value, action_dist = self.head(hidden_features)

        if not deterministic:
            action = action_dist.sample() # this might need to be adapted if we have to use 2 Gaussians in the head
        else:
            action = action_dist.probs.argmax(dim=-1, keepdim=True) ##TODO: this does not work

        action_log_probs = action_dist.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)

        return value, action, action_log_probs

    def forward(inputs):
        raise NotImplementedError

    def get_value(self, inputs):
        """
        Given the inputs returns the predicted value
        """
        hidden_features = self.base(inputs)
        value, _ = self.head(hidden_features)

        return value

    def evaluate_actions(self, inputs, action):
        """
        Given an input returns the values, action log probabilities and entropies required for the update
        """
        hidden_features = self.base(inputs)
        value, action_dist = self.head(hidden_features)

        action_log_probs = action_dist.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy

class Base(nn.Module):
    def __init__(self, hidden_size):
        super(Base, self).__init__()
        self.output_size = hidden_size

    def forward(self, inputs):
        raise NotImplementedError

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SimpleCNNBase(Base):
    """
    Simple policy base for image inputs to policy
    """
    def __init__(self, num_inputs, hidden_size=32 * 7 * 7, use_grid=False):
        super(SimpleCNNBase, self).__init__(hidden_size)
        self.use_grid = use_grid
        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten(),
        )


    def forward(self, inputs):
        """
        Encodes the inputs to the policy into a latent state and returns it
        """
        def create_coord_buffer(patch_shape):
            ys = torch.linspace(-1, 1, patch_shape[0])
            xs = torch.linspace(-1, 1, patch_shape[1])
            xv, yv = torch.meshgrid(ys, xs)
            coord_map = torch.stack((xv, yv)).unsqueeze(0)
            return coord_map

        inp = (inputs/255.)

        if self.use_grid:
            coord_conv = create_coord_buffer(inputs.size()[-2:])
            inp = torch.cat([inp, coord_conv.to('cuda').expand(inp.size()[0], -1, -1, -1)], 1)

        x = self.main(inp)

        return x


class SimpleControlBase(Base):
    """
    Policy base for control inputs; testing for gym
    """
    def __init__(self, num_inputs, hidden_size=32):
        super(SimpleControlBase, self).__init__(hidden_size)
        self.main = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )

    def forward(self, inputs):
        x = self.main(inputs)
        return x


class ContinuousActionHead(Head):
    """
    Policy head for continuous action outputs
    """
    def __init__(self, num_inputs, action_space):
        super(ContinuousActionHead, self).__init__()

    def forward(self, inputs):
        """
        Gets input from base part of the policy and returns the value and action distribution
        reference could be: https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_heads.py
            GaussianActorCriticNet
        """
        pass


class DiscreteActionHead(Head):
    """
    Policy head for discrete action outputs
    """
    def __init__(self, num_inputs, action_space, hidden_size, use_deep_layers=False):
        super(DiscreteActionHead, self).__init__()
        self.use_deep_layers = use_deep_layers

        self.main = nn.Linear(num_inputs, hidden_size)
        self.in_between = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                )
        self.values_head = nn.Linear(hidden_size, 1)
        self.actor_head = nn.Linear(hidden_size, action_space)

    def forward(self, inputs):
        """
        Gets input from base part of the policy and returns the value and action distribution
        """
        x = F.relu(self.main(inputs))

        if self.use_deep_layers:
            x = self.in_between(x)

        return self.values_head(x), torch.distributions.Categorical(logits = self.actor_head(x))

