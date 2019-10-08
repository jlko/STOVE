import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    def __init__(self, actor_critic):
        super(Agent, self).__init__()

    def update(self):
        raise NotImplementedError


class PPOAgent(Agent):
    """
    Reinforcement learning agent based on proximal policy optimization.
    """
    def __init__(self,
                actor_critic,
                clip_param,
                ppo_epoch,
                num_mini_batch,
                value_loss_coef,
                entropy_coef,
                lr=None,
                eps=None,
                max_grad_norm=None,
                use_clipped_value_loss=True):
        super(PPOAgent, self).__init__(actor_critic)

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        # TODO: uncomment this as soon as there is parameters in the network

    def update(self, obs, actions, returns, values, action_log_probs, adv_targ):
        """
        Uses ppo_epoch number of updates using num_mini_batches of proximal policy to update the actor_critic
        Inputs should be shuffled and flattened already

        Args:
            obs (torch.ByteTensor), shape=(b * n, res, res, num_ch)
            actions (torch.), shape=(b * n, 1)  
            returns (torch.FloatTensor), shape=(b * n, 1)
            values (torch.FloatTensor), shape= (b * n, 1)
            action_log_probs (torch.FloatTensor), shape= (b * n, 1) 

        Returns:
            value_loss_epoch (Float): average value loss across all ppo_epochs
            action_loss_epoch (Float): average action loss across all ppo_epochs
            dist_entropie_epoch (Float): average dist entropy across all ppo_epochs

        """
        advantages = returns - values 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        mb_size = int(obs.size()[0] / self.num_mini_batch)

        for e in range(self.ppo_epoch):
            random_permutations = torch.randperm(obs.size()[0])
            _obs = obs[random_permutations]
            _actions = actions[random_permutations]
            _returns = returns[random_permutations]
            _values = values[random_permutations]
            _action_log_probs = action_log_probs[random_permutations]
            _adv_targ = adv_targ[random_permutations]

            for i in range(self.num_mini_batch):
                mb_obs = _obs[i * mb_size : (i+1) * mb_size]
                mb_actions = _actions[i * mb_size : (i+1) * mb_size]
                mb_values = _values[i * mb_size : (i+1) * mb_size]
                mb_returns = _returns[i * mb_size : (i+1) * mb_size]
                mb_action_log_probs = _action_log_probs[i * mb_size : (i+1) * mb_size]
                mb_adv_targ = _adv_targ[i * mb_size : (i+1) * mb_size]

                new_values, new_action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(mb_obs, mb_actions)

                # Compute action loss
                ratio = torch.exp(new_action_log_probs -
                                    mb_action_log_probs)
                surr1 = ratio * mb_adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * mb_adv_targ
                
                action_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_pred_clipped = mb_values + (new_values - mb_values).clamp(-self.clip_param, self.clip_param)
                value_losses = (new_values - mb_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Do optimization step
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                    dist_entropy * self.entropy_coef).backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                            self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

