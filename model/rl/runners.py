import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from agents import PPOAgent
from rl_model import Policy

# TODO: this is ugly as hell but python sucks sometimes, should try to put everything in packages?
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from billards_env import BillardsEnv, AvoidanceTask


class Runner():
    def __init__(
        self,
        env,
        task,
        summary_path,
        device,
        agent,
        actor_critic,
        num_steps,
        batch_size,
        discount,
        do_train):
        super(Runner, self).__init__()

        self.env = env
        self.task = task
        self.summary_path = summary_path
        self.device = device
        self.agent = agent
        self.actor_critic = actor_critic
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.discount = discount
        self.do_train = do_train
        self.summary_writer = SummaryWriter(self.summary_path)

        self.batch_counter = 0
        self.episode_counter = 0

        self.actor_critic.to(self.device)

    def _calculate_returns(self, next_value, rewards, dones):
        """
        Calculate the discounted returns

        Args:
            rewards (torch.FloatTensor), shape=(b, n, 1):
                rewards obtained during the last trajectory
            next_value (torch.FloatTensor), shape=(b, 1):
                value of the next state at (num_steps+1)
        Returns:
            returns (torch.FloatTensor), shape=(b, n, 1):
                discounted returns for each step in the trajectory
        """
        returns = torch.zeros((rewards.size()[0], rewards.size()[1] + 1, 1)).to(self.device)
        returns[:, -1] = next_value
        for step in reversed(range(rewards.size()[1])):
            returns[:, step] = returns[:, step+1] * self.discount * (1 - dones[:, step]) + rewards[:, step]

        return returns[:, :-1]

    def _log_to_tb(self, log_dict, use_total_steps=True):
        """
        Iterate over input dictionary and add all values to tensorboard under the
        name of the key

        Args:
            log_dict (dict(str: float))
        """
        if use_total_steps:
            counter = self.batch_counter * self.num_steps * self.batch_size
        else:
            counter = self.batch_counter
        for key, value in log_dict.items():
            self.summary_writer.add_scalar(key, value, counter)

    def reset(self, frame_buffer=False):
        """
        Resets the environment in the Runner
        Returns:

        """
        # TODO: I think we do not have a reset method in our env yet
        self.env.reset()
        if frame_buffer:
            self.task.frame_buffer[:] = 0.
            for i in range(3):
                self.task.step_frame_buffer(0)
            return self.task.step_frame_buffer(0)
        else:
            img, state, _, _ = self.task.step(0)
            return img, state

    def run_batch(self):
        raise NotImplementedError


class PPORunner(Runner):
    def __init__(
            self,
            env,
            task,
            summary_path,
            device,
            agent,
            actor_critic,
            num_steps=128,
            batch_size=32,
            discount=0.99,
            do_train=True):
        super(PPORunner, self).__init__(env, task, summary_path, device, agent, actor_critic, num_steps, batch_size, discount, do_train)

    def run_batch(self):
        """
        Run a batch of num_steps and update the policy using the batch_data
        """
        storage = BatchStorage(self.batch_size, self.num_steps, self.task.get_framebuffer_shape(), self.device)

        for b in range(self.batch_size):
            initial_obs, _, _, _ = self.reset(frame_buffer=True)
            initial_obs = torch.tensor(initial_obs, requires_grad=False).to(self.device)
            storage.obs[b, 0] = initial_obs

            for step in range(self.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(storage.obs[b, step:step+1].permute(0, 3, 1, 2))
                img, state, r, done = self.task.step_frame_buffer(action[0].detach().cpu().numpy())

                storage.insert(b, step, img, action, r, value, done, action_log_prob)

        next_values = self.actor_critic.get_value(storage.obs[:, -1].permute(0, 3, 1, 2)).detach()
        returns = self._calculate_returns(next_values, storage.rewards, storage.dones).detach()
        adv_targ = (returns - storage.values).detach()
        storage.obs = storage.obs[:, :-1]

        value_loss, action_loss, dist_entropy = self._train_ppo(storage.obs.permute(0, 1, 4, 2, 3), storage.actions, returns, storage.values, storage.action_log_probs, adv_targ)

        log_dict = {
            '/losses/value_loss': value_loss,
            '/losses/action_loss': action_loss,
            '/losses/entropy':  dist_entropy,
            '/info/rewards': np.mean(storage.rewards.cpu().numpy())
        }

        self._log_to_tb(log_dict)
        self.batch_counter += 1

    def run_batch_states(self):
        """
        Run a batch of num_steps and update the policy using the batch_data
        """
        storage = BatchStorage(self.batch_size, self.num_steps, (self.env.get_state_shape()[0] * self.env.get_state_shape()[1],), self.device)


        for b in range(self.batch_size):
            _, initial_state = self.reset()
            initial_state = np.reshape(initial_state, (1, -1))
            initial_state = torch.tensor(initial_state, requires_grad=False).to(self.device) 
            storage.obs[b, 0] = initial_state

            for step in range(self.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(storage.obs[b, step:step+1])
                img, state, r, done = self.task.step_frame_buffer(action[0].detach().cpu().numpy())
                state = state.reshape(state.shape[0] * state.shape[1])

                storage.insert(b, step, state, action, r, value, done, action_log_prob)
        
        next_values = self.actor_critic.get_value(storage.obs[:, -1]).detach()
        returns = self._calculate_returns(next_values, storage.rewards, storage.dones).detach()
        adv_targ = (returns - storage.values).detach()
        storage.obs = storage.obs[:, :-1]

        value_loss, action_loss, dist_entropy = self._train_ppo(storage.obs, storage.actions, returns, storage.values, storage.action_log_probs, adv_targ)

        log_dict = {
            '/losses/value_loss': value_loss,
            '/losses/action_loss': action_loss,
            '/losses/entropy':  dist_entropy,
            '/info/rewards': np.mean(storage.rewards.cpu().numpy())
        }

        self._log_to_tb(log_dict)
        self.batch_counter += 1

    def run_gym_batch(self):
        """
        Run a batch of num_steps and update the policy using the batch_data
        """
        storage = BatchStorage(self.batch_size, self.num_steps, (*self.env.observation_space.shape,), self.device)

        for b in range(self.batch_size):
            initial_obs = self.env.reset()
            storage.obs[b, 0] = torch.tensor(initial_obs, requires_grad=False).to(self.device)

            for step in range(self.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(storage.obs[b, step:step+1])
                state, r, done, info = self.env.step(action[0].detach().cpu().numpy())

                storage.insert(b, step, state, action, r, value, done, action_log_prob)

        next_values = self.actor_critic.get_value(storage.obs[:, -1]).detach()
        returns = self._calculate_returns(next_values, storage.rewards, storage.dones).detach()
        adv_targ = (returns - storage.values).detach()
        storage.obs = storage.obs[:, :-1]

        value_loss, action_loss, dist_entropy = self._train_ppo(storage.obs, storage.actions, returns, storage.values, storage.action_log_probs, adv_targ)

        log_dict = {
            '/losses/value_loss': value_loss,
            '/losses/action_loss': action_loss,
            '/losses/entropy':  dist_entropy,
            '/info/dyna_rewards': np.mean(storage.rewards.cpu().numpy())
        }

        self._log_to_tb(log_dict)
        self.batch_counter += 1

    def run_pretrained_batch(self, state_model, encoded_states, apps):
        
        def _sample_from_states(batch_size):
            randints = torch.randint(0, len(encoded_states) - 1, (batch_size,)).long()
            return encoded_states[randints].clone(), apps[randints].clone()

        storage = BatchStorage(self.batch_size, self.num_steps, encoded_states.shape[1:], self.device)#
        storage.obs[:, 0], appearances = _sample_from_states(self.batch_size)
        
        for step in range(self.num_steps):
            full_model_state = storage.obs[:, step]
            current_physical_state = full_model_state[:, :, 2:6].data.clone()
            current_physical_state[:, :, :2] = ((current_physical_state[:, :, :2] + 1) / 2) * self.env.hw
            current_physical_state[:, :, 2:] = current_physical_state[:, :, 2:] / 2 * self.env.hw
            with torch.no_grad():
                value, action, action_log_prob = self.actor_critic.act(current_physical_state.reshape((self.batch_size, -1)))
                one_hot_actions = F.one_hot(action.long(), self.task.get_action_space()).unsqueeze(dim=1).float().to(self.device)
               # one_hot_actions = torch.eye(self.task.get_action_space())[action.long()].to(self.device)
                full_model_state, r = state_model.rollout(full_model_state, 1, actions=one_hot_actions, appearance=appearances) 
            done = torch.zeros((self.batch_size, 1))
            storage.insert_batch(step, full_model_state[:, 0], action.unsqueeze(1), r[:, 0] - 1 , value, done, action_log_prob)

        storage.obs[:, :, :, 2:4] = (storage.obs[:, :, :, 2:4] + 1) / 2 * self.env.hw
        storage.obs[:, :, :, 4:6] = storage.obs[:, :, :, 4:6] / 2 * self.env.hw

        next_values = self.actor_critic.get_value(storage.obs[:, -1, :, 2:6].reshape((self.batch_size, -1))).detach()
        returns = self._calculate_returns(next_values, storage.rewards, storage.dones).detach()
        adv_targ = (returns - storage.values).detach()
        storage.obs = storage.obs[:, :-1]
        value_loss, action_loss, dist_entropy = self._train_ppo(storage.obs[:, :, :, 2:6].reshape(self.batch_size, self.num_steps, -1), 
                                                                storage.actions, returns, storage.values, storage.action_log_probs, adv_targ)

        test_reward = self._test_actorcritic(16, 32)
        
        log_dict = {
            '/losses/value_loss': value_loss,
            '/losses/action_loss': action_loss,
            '/losses/entropy':  dist_entropy,
        }

        self._log_to_tb(log_dict)
        self.batch_counter += 1

    def _test_actorcritic_images(self, batch_size, num_steps):
        test_storage = BatchStorage(batch_size, num_steps, self.task.get_framebuffer_shape(), self.device)

        for b in range(batch_size):
            initial_obs, _, _, _ = self.reset(frame_buffer=True)
            initial_obs = torch.tensor(initial_obs, requires_grad=False).to(self.device)
            test_storage.obs[b, 0] = initial_obs

            for step in range(num_steps):
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(test_storage.obs[b, step:step+1].permute(0, 3, 1, 2))
                img, state, r, done = self.task.step_frame_buffer(action[0].detach().cpu().numpy())

                test_storage.insert(b, step, img, action, r, value, done, action_log_prob)

        mean_reward = np.mean(test_storage.rewards.cpu().numpy())
        return mean_reward

    def _test_actorcritic(self, batch_size, num_steps):
        test_storage = BatchStorage(batch_size, num_steps, (self.env.get_state_shape()[0] * self.env.get_state_shape()[1],), self.device)

        for b in range(batch_size):
            _, initial_state = self.reset()
            initial_state = np.reshape(initial_state, (1, -1))
            initial_state = torch.tensor(initial_state, requires_grad=False).to(self.device) 
            test_storage.obs[b, 0] = initial_state

            for step in range(num_steps):
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(test_storage.obs[b, step:step+1])
                img, state, r, done = self.task.step(action[0].detach().cpu().numpy())
                state = state.reshape(state.shape[0] * state.shape[1])

                test_storage.insert(b, step, state, action, r, value, done, action_log_prob)

        mean_reward = np.mean(test_storage.rewards.cpu().numpy())
        return mean_reward

    def _train_ppo(self, obs, actions, returns, values, action_log_probs, adv_targ):
        """
        Given the required inputs for PPO flattens the batch and runs PPO, returns average loss values
        """

        def _flatten_first_dims(inp):
            return inp.reshape((-1 , *inp.size()[2:]))

        obs = _flatten_first_dims(obs)
        actions = _flatten_first_dims(actions)
        returns = _flatten_first_dims(returns)
        values = _flatten_first_dims(values)
        action_log_probs = _flatten_first_dims(action_log_probs)
        adv_targ = _flatten_first_dims(adv_targ)

        value_loss, action_loss, dist_entropy = self.agent.update(obs, actions, returns, values, action_log_probs, adv_targ)

        return value_loss, action_loss, dist_entropy

class BatchStorage():
    def __init__(self, batch_size, num_steps, obs_shape, device):
        super(BatchStorage, self).__init__()
        self.obs = torch.zeros((batch_size, num_steps + 1, *obs_shape)).to(device)
        self.actions = torch.zeros((batch_size, num_steps, 1)).to(device)
        self.rewards = torch.zeros((batch_size, num_steps, 1)).to(device)
        self.values = torch.zeros((batch_size, num_steps, 1)).to(device)
        self.dones = torch.zeros((batch_size, num_steps, 1)).to(device)
        self.action_log_probs = torch.zeros((batch_size, num_steps, 1)).to(device)

    def insert(self, b, step, o, a, r, v, d, alp):
        self.obs[b, step + 1] = torch.tensor(o, requires_grad=False)
        self.rewards[b, step] = torch.tensor(r, requires_grad=False)
        self.dones[b, step] = torch.tensor(d, requires_grad=False)

        self.actions[b, step] = a
        self.values[b, step] = v
        self.action_log_probs[b, step] = alp

    def insert_batch(self, step, o, a, r, v, d, alp):
        self.obs[:, step + 1] = o
        self.rewards[:, step] = r
        self.dones[:, step] = d
        self.actions[:, step] = a
        self.values[:, step] = v
        self.action_log_probs[:, step] = alp

def test_run():
    """
    use this to test runner
    """
    env = BillardsEnv()
    task = AvoidanceTask(env)

    actor_critic = Policy(env.get_obs_shape(), task.get_action_space()) # TODO: get action space not yet implemented in env
    agent = PPOAgent(actor_critic, clip_param=0.1,
                     ppo_epoch=4, num_mini_batch=32,
                     value_loss_coef=0.5, entropy_coef=0.01,
                     lr=2e-4, eps=1e-5, max_grad_norm=40)

    runner = PPORunner(env=env, task=task, device='cuda', summary_path='./summary/5', agent=agent, actor_critic=actor_critic)

    num_batches = 2000

    try:
        for i in range(num_batches):
            runner.run_batch()

            if i % 100 == 0:
                torch.save(actor_critic, './models/model_5_' + str(i) + '.pt' )

    except:
        torch.save(actor_critic, './models/model_5_crash.pt')
        exit(0)

    torch.save(actor_critic, './models/model_5_end.pt')

if __name__ == "__main__":
    test_run()
