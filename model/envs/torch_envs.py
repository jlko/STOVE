import numpy as np
import scipy as sc
from scipy import integrate
from scipy.io import savemat

import torch

import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm

import pickle

def norm(x):
    if len(x.shape) == 1:
        _norm = torch.norm(x)
    else:
        _norm = torch.norm(x, dim=1).view(-1, 1)
    return _norm


class Task():
    # action_selection = [
    #         np.array([0., 0.]),
    #         np.array([1., 0.]),
    #         np.array([0., 1.]),
    #         np.array([1., 1.]),
    #         np.array([-1., 0.]),
    #         np.array([0., -1.]),
    #         np.array([-1., -1.]),
    #         np.array([-1., 1.]),
    #         np.array([1., -1.]),
    #         ]
    # action_force = 0.1

    def __init__(self, env):
        self.env = env
        self.frame_buffer = np.zeros((*env.get_obs_shape()[:2], env.get_obs_shape()[2] * 4))

#    def action(self, _action):
#        _action = Task.action_selection[_action] * Task.action_force
#        return _action

    def get_action_space(self):
        return 3 #len(Task.action_selection)

    def calculate_reward(self, state, action, env):
        raise NotImplementedError

    def step(self, _action):
        img, state, done = self.env.step(_action)
        r = self.calculate_reward(state, _action, self.env)
        return img, state, r, done

    def step_frame_buffer(self, _action):
        img, state, done = self.env.step(_action)
        r = self.calculate_reward(state, _action, self.env)

        num_channels = self.env.get_obs_shape()[2]
        self.frame_buffer[:, :, :-num_channels] = self.frame_buffer[:, :, num_channels:]
        self.frame_buffer[:, :, -num_channels:] = img

        return self.frame_buffer, state, r, done


class AvoidanceTask(Task):

    def __init__(self, env, dist=.1):
        super().__init__(env)
        self.dist = dist

    def calculate_reward(self, state, action, env):
        r = 0
        for i in range(1, self.env.n):
            if norm(state[i, 0:2] - state[0, 0:2]) - (env.r[0] + env.r[i]) < self.dist:
                r -= 1
        return r


class MaxDistanceTask(Task):

    def calculate_reward(self, state, action, env):
        r = 0
        for i in range(1, self.env.n):
            r += norm(state[i, 0:2] - state[0, 0:2])
        return r


class MinDistanceTask(Task):
    def calculate_reward(self, state, action, env):
        return -MaxDistanceTask(self.env).calculate_reward(state, action, env)


class PhysicsEnv():
    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1., init_v_factor=0, friction_coefficient=0., device='cpu'):
        """
        Initializes a physics env with some general parameters

        Args:
            n (int): optional, Number of objects in the scene
            r (float)/list(float): optional, Radius of objects in the scene (used for rendering and some environments)
            m (float)/list(float): optional, mass of the objects in the scene
            hw (float): optional, wall distance
            eps (float): optional, internal simulation granularity as a fraction of one time step
            res (int): optional, resultion of the output picture
        """
        self.device = device
        self.n = n
        self.r = torch.Tensor([[r]] * n).to(self.device) if np.isscalar(r) else r
        self.m = torch.Tensor([[m]] * n).to(self.device) if np.isscalar(m) else m
        self.v = torch.randn((self.n, 2)).to(self.device)
        self.v = self.v / torch.sqrt((self.v**2).sum()) * .5
        self.a = torch.zeros_like(self.v).to(self.device)
        self.hw = hw
        self.eps = t/granularity
        self.internal_steps = granularity
        self.res = res
        self.t = t
        self.x = self.init_x()
        self.fric_coeff = friction_coefficient
        self.v_rotation_angle = 2 * 3.14159 * 0.02

        self.use_colors = False

    def init_x(self):
        """
        Initializes the env with no overlap between the objects and walls
        """
        good_config = False
        while not good_config:
            x = torch.rand(self.n, 2).to(self.device) * self.hw/2 + self.hw/4
            good_config = True
            for i in range(self.n):
                for z in range(2):
                    if x[i][z] - self.r[i] < 0:
                        good_config = False
                    if x[i][z] + self.r[i] > self.hw:
                        good_config = False

            for i in range(self.n):
                for j in range(i):
                    if norm(x[i] - x[j]) < self.r[i] + self.r[j]:
                        good_config = False
        return x

    def get_rot(self, action):
        direction = torch.Tensor([(action - 1) * self.eps])
        return torch.Tensor([[torch.cos(-self.v_rotation_angle * direction), -torch.sin(-self.v_rotation_angle * direction)],
                       [torch.sin(-self.v_rotation_angle * direction), torch.cos(-self.v_rotation_angle * direction)]]).to(self.device)

    def simulate_physics(self):
        """
        Calculates a single time step in the simulation at fine granularity. Controls
        the physics of the environment

        Args:
            action (np.Array(float)): a two dimensional float giving an x,y force to
                                      enact upon the first object

        Returns:
            d_vs (np.Array(float)): velocity updates for the simulation
        """
        raise NotImplementedError

    def step(self, action=1, mass_center_obs=False):
        for i in range(self.internal_steps):

            self.x += self.t * self.eps * self.v

            if mass_center_obs:
                c_body = torch.sum(self.m * self.x, 0) / torch.sum(self.m)
                self.x += self.hw/2 - c_body

            #self.v[0] += (action/self.m[0]) * self.t * self.eps
            self.v[0] = torch.matmul(self.get_rot(action), self.v[0])
            self.v -= self.fric_coeff * self.m * self.v * self.t * self.eps

            self.v = self.simulate_physics()

        img = self.draw_image()
        state = torch.cat([self.x, self.v], dim=1)
        done = False
        return img, state, done

    def get_obs_shape(self):
        return (self.res, self.res, 3)

    def ar(self, x, y, z):
        return z / 2 + torch.arange(x, y, z).to(self.device)

    def draw_image(self):
        if self.n > 3 and not self.use_colors or self.n > 6:
            raise IllegalArgumentException('Must use color palette when visualizing more than 3 balls and can only view max 6')
        img = torch.zeros((self.res, self.res, 3)).to(self.device)
        [I, J] = torch.meshgrid(self.ar(0, 1, 1. / self.res) * self.hw,
                             self.ar(0, 1, 1. / self.res) * self.hw)

        colors = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]]).to(self.device)

        for i in range(self.n):
            factor = torch.exp(- (((I - self.x[i, 0]) ** 2 +
                (J - self.x[i, 1]) ** 2) /
                (self.r[i] ** 2)) ** 4)
            if self.use_colors:
                img[:, :, 0] += colors[i, 0] * factor
                img[:, :, 1] += colors[i, 1] * factor
                img[:, :, 2] += colors[i, 2] * factor
            else:
                img[:, :, i] += factor

        img[img > 1] = 1
        return img

    def reset(self):
        """
        Resets the environment to a new configuration
        """
        self.v = torch.randn((self.n, 2)).to(self.device)
        self.v = self.v / norm(self.v) * .5
        self.a = torch.zeros_like(self.v).to(self.device)
        self.x = self.init_x()


class BillardsEnv(PhysicsEnv):

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1., init_v_factor=0, friction_coefficient=0.):
        super().__init__(n, r, m, hw, granularity, res, t, init_v_factor, friction_coefficient)
        self.oob = 3*[False]

    def simulate_physics(self):
        # F = ma = m dv/dt ---> dv = a * dt = F/m * dt

        v = self.v.clone()

        # main loop:
        #     if wall_collision and not oob:
        #     # collide with wall
        #         self.oob[i] = True
        #     if not wall_collision and oob:
        #         self.oob[i] = False

        # check for collisions with wall
        for i in range(self.n):
            for z in range(2):
                if not self.r[i] < self.x[i, z] + (v[i, z] * self.eps * self.t):
                    self.x[i, z] = self.r[i]
                    v[i, z] = - v[i, z]
                elif not self.x[i, z] + (v[i, z] * self.eps * self.t) < (self.hw - self.r[i]):
                    self.x[i, z] = self.hw - self.r[i]
                    v[i, z] = - v[i, z]

        # check for collisions with objects
        for i in range(self.n):
            for j in range(i):
                if norm(self.x[i] - self.x[j]) < (self.r[i] + self.r[j]):
                    w = self.x[i] - self.x[j]
                    w = w / norm(w)
                    v_i = torch.dot(w, v[i])
                    v_j = torch.dot(w, v[j])

                    new_v_i, new_v_j = self.new_speeds(self.m[i], self.m[j], v_i, v_j)

                    v[i] += w * (new_v_i - v_i)
                    v[j] += w * (new_v_j - v_j)

        return v

    def new_speeds(self, m1, m2, v1, v2):
        new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
        new_v1 = new_v2 + (v2 - v1)
        return new_v1, new_v2


class GravityEnv(PhysicsEnv):

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1, init_v_factor=0.2):
        super().__init__(n, r, m, hw, granularity, res)
        self.G = 0.5
        self.K1 = self.G
        self.K2 = 1
        self.use_colors = True
        self.init_v(init_v_factor)

    def init_x(self):
        """
        Initializes the env with no overlap between the objects and walls
        """
        good_config = False
        counter = 0
        while not good_config and counter < 1000:
            x = torch.rand(self.n, 2) * self.hw/3 + self.hw/3
            good_config = True
            for i in range(self.n):
                for j in range(i):
                    good_config = good_config and norm(x[i] - x[j]) > 5
            counter += 1
        return x

    def init_v(self, factor):
        x_middle = torch.sum(self.x, 0) / self.n
        for i in range(self.n):
            v = - (x_middle - self.x[i])
            v = (v/norm(v)) * factor
            self.v[i] = [v[1], -v[0]]

    def step(self, action):
        return super().step(action, True)

    def simulate_physics(self):
        x_middle = torch.array([self.hw/2, self.hw/2])
        v = torch.zeros_like(self.v)
        for i in range(self.n):
            F_tot = torch.Tensor([0., 0.]).to(self.device)
            for j in range(self.n):
                if i != j:
                    r = torch.norm(self.x[j] - self.x[i])
                    F_tot -= self.G * self.m[j] * self.m[i] * (self.x[i] - self.x[j]) / ((r + 1e-5) ** 3)
            r = (x_middle - self.x[i])
            F_tot += 0.001 * (r ** 3) / norm(r)
            F_tot = torch.clip(F_tot, -1, 1)
            v[i] = self.v[i] + (F_tot/self.m[i]) * self.t * self.eps
        return v


def generate_fitting_run(env_class, run_len=100, run_num=1000, max_tries=10000,
                         res=50, n=2, r=1., dt=0.01, gran=10):
    m = 1.
    init_v = [0.1, 0.2, 0.3, 0.5, 1.]

    good_counter = 0
    bad_counter = 0
    good_imgs = []
    good_states = []

    for _try in tqdm(range(max_tries)):
        _init_v = np.random.choice(init_v)
        # init_v is ignored for BillardsEnv
        env = env_class(n=n, m=m, granularity=gran, r=r, t=dt, hw=10, res=res, init_v_factor=_init_v, friction_coefficient=0.3)
        run_value = 0

        all_imgs = np.zeros((run_len, *env.get_obs_shape()))
        all_states = np.zeros((run_len, env.n, 4))

        run_value = 0
        for t in tqdm(range(run_len)):
            img, state, done = env.step(0)
            all_imgs[t] = img
            all_states[t] = state

            run_value += np.sum(np.logical_and(0 < state[:, :2], state[:, :2] < env.hw)) / (env.n * 2)

        if run_value > (run_len - run_len/100):
            good_imgs.append(all_imgs)
            good_states.append(all_states)
            good_counter += 1
        else:
            bad_counter += 1

        if good_counter >= run_num:
            break

    good_imgs = np.stack(good_imgs, 0)
    good_states = np.stack(good_states, 0)

    print('Generation of {} runs finished, total amount of bad runs: {}'.format(
        run_num, bad_counter))

    return good_imgs, good_states


def main():
    r = 1.  # np.array([[1.], [1.], [1.]])
    m = 1.  # np.pi * (r ** 2)
    env = BillardsEnv(n=3, m=m, r=r, granularity=10, t=1., hw=10, res=32, friction_coefficient=0.)
    task = MaxDistanceTask(env)
    T = 1000
    all_imgs = np.zeros((T, *env.get_obs_shape()))

    for t in tqdm(range(T)):
        img, state, reward, done = task.step(1)
        if reward < 0:
            img = 1 - img
        all_imgs[t] = img

    all_imgs = (255 * all_imgs).astype(np.uint8)
    imageio.mimsave('./output.gif', all_imgs, fps=24)


def generate_billards_data():

    for run_types, run_num in zip(['train', 'test'], [1000, 300]):
        X, y = generate_fitting_run(BillardsEnv, res=32, r=1.2, n=3, dt=1., gran=2, run_num=run_num)
        data = dict()
        data['X'] = X
        data['y'] = y
        savemat('./data/billards_new_env_{}_data.mat'.format(run_types), data)

    first_seq = (255 * X[1]).astype(np.uint8)
    imageio.mimsave('./data/billards_new_env.gif', first_seq, fps=24)

if __name__ == '__main__':
    main()
    # generate_fitting_run(GravityEnv)
    # generate_billards_data()
