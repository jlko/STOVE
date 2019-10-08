import argparse
import pickle
import imageio
import numpy as np
import scipy as sc
from scipy import integrate
from scipy.io import savemat
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite


def norm(x):
    if len(x.shape) == 1:
        _norm = np.linalg.norm(x)
    else:
        _norm = np.linalg.norm(x, axis=1).reshape(-1, 1)
    return _norm


class Task():
    angular = 1./np.sqrt(2)
    action_selection = [
            np.array([0., 0.]),
            np.array([1., 0.]),
            np.array([0., 1.]),
            np.array([angular, angular]),
            np.array([-1., 0.]),
            np.array([0., -1.]),
            np.array([-angular, -angular]),
            np.array([-angular, angular]),
            np.array([angular, -angular]),
            ]

    def __init__(self, env, num_stacked=4, greyscale=False, action_force=.3):
        self.env = env
        self.env.m[0] = 10000
        if greyscale:
            self.frame_buffer = np.zeros((*env.get_obs_shape()[:2], num_stacked))
            self.conversion = lambda x: np.sum(x * [[[0.3, 0.59, 0.11]]], 2, keepdims=True)
        else:
            self.frame_buffer = np.zeros((*env.get_obs_shape()[:2], env.get_obs_shape()[2] * num_stacked))
            self.conversion = lambda x: x
        self.frame_channels = 3 if not greyscale else 1
        self.action_force = action_force

    def get_action_space(self):
        return 9 #len(Task.action_selection)

    def get_framebuffer_shape(self):
        return self.frame_buffer.shape

    def calculate_reward(self, state, action, env):
        raise NotImplementedError

    def resolve_action(self, _action, env=None):
        # change this to change action
        action = self.action_selection[_action]
        action = action * self.action_force
        return action

    def step(self, _action):
        action = self.resolve_action(_action)
        img, state, done = self.env.step(action, actions=True)
        r = self.calculate_reward(state, action, self.env)
        return img, state, r, done

    def step_frame_buffer(self, _action):
        action = self.resolve_action(_action)
        img, state, done = self.env.step(action, actions=True)
        r = self.calculate_reward(state, action, self.env)

        img = self.conversion(img)
        self.frame_buffer[:, :, :-self.frame_channels] = self.frame_buffer[:, :, self.frame_channels:]
        self.frame_buffer[:, :, -self.frame_channels:] = img

        return self.frame_buffer, state, r, done


class AvoidanceTask(Task):

    def calculate_reward(self, state, action, env):
        return -env.collisions


class MaxDistanceTask(Task):

    def calculate_reward(self, state, action, env):
        r = 0
        for i in range(1, self.env.n):
            scaling = 2
            current_norm = norm(state[i, 0:2] - state[0, 0:2]) - 2 * env.r[0]
            current_exp = -np.clip(np.exp(-current_norm * scaling), 0, 1)
            r = min(r, current_exp)
        return r


class MinDistanceTask(Task):
    def calculate_reward(self, state, action, env):
        r = - ((100 * env.hw) ** 2)
        for i in range(1, env.n):
            r = max(r, -(norm(state[i, 0:2] - state[0, 0:2]) - 2 * env.r[0]) ** 2)
        return r


class PhysicsEnv():
    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1.,
                 init_v_factor=0, friction_coefficient=0., seed=None, sprites=False):
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
        np.random.seed(seed)
        self.n = n
        self.r = np.array([[r]] * n) if np.isscalar(r) else r
        self.m = np.array([[m]] * n) if np.isscalar(m) else m
        self.v = np.random.normal(size=(self.n, 2))
        self.v = self.v / np.sqrt((self.v**2).sum()) * .5
        self.a = np.zeros_like(self.v)
        self.hw = hw
        self.eps = 1/granularity
        self.internal_steps = granularity
        self.res = res
        self.t = t
        self.x = self.init_x()
        self.fric_coeff = friction_coefficient
        self.v_rotation_angle = 2 * np.pi * 0.05

        self.use_colors = False

        if sprites:
            self.renderer = spriteworld_renderers.PILRenderer(
                image_size=(self.res, self.res),
                anti_aliasing=10,
                )

            shapes = ['triangle', 'square', 'circle', 'star_4']

            if not np.isscalar(r):
                print("Scale elements according to radius of first element.")

            # empirical scaling rule, works for r = 1.2 and 2
            self.scale = self.r[0]/self.hw / 0.6
            self.shapes = np.random.choice(shapes, 3)
            if not np.isscalar(r):
                print("Scale elements according to radius of first element.")
            # empirical scaling rule, works for r = 1.2 and 2
            self.scale = self.r[0]/self.hw / 0.6
            self.draw_image = self.draw_sprites
        else:
            self.draw_image = self.draw_balls


    def init_x(self):
        """
        Initializes the env with no overlap between the objects and walls
        """
        good_config = False
        while not good_config:
            x = np.random.rand(self.n, 2) * self.hw/2 + self.hw/4
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
        direction = (action - 1) * self.eps
        return np.array([[np.cos(-self.v_rotation_angle * direction), -np.sin(-self.v_rotation_angle * direction)],
                        [np.sin(-self.v_rotation_angle * direction), np.cos(-self.v_rotation_angle * direction)]])

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

    def step(self, action=1, mass_center_obs=False, actions=False):
        if actions:
            self.v[0] = action * self.t
        for i in range(self.internal_steps):
            self.x += self.t * self.eps * self.v

            if mass_center_obs:
                c_body = np.sum(self.m * self.x, 0) / np.sum(self.m)
                self.x += self.hw/2 - c_body

            #self.v[0] += (action/self.m[0]) * self.t * self.eps
            #self.v[0] = np.matmul(self.get_rot(action), self.v[0])
            self.v -= self.fric_coeff * self.m * self.v * self.t * self.eps
            self.v = self.simulate_physics(actions=actions)

        img = self.draw_image()
        state = np.concatenate([self.x, self.v], axis=1)
        done = False
        return img, state, done

    def get_obs_shape(self):
        return (self.res, self.res, 3)

    def get_state_shape(self):
        state = np.concatenate([self.x, self.v], axis=1)
        return state.shape

    def ar(self, x, y, z):
        return z / 2 + np.arange(x, y, z, dtype='float')

    def draw_balls(self):
        if self.n > 3 and not self.use_colors or self.n > 6:
            raise IllegalArgumentException('Must use color palette when visualizing more than 3 balls and can only view max 6')
        img = np.zeros((self.res, self.res, 3), dtype='float')
        [I, J] = np.meshgrid(self.ar(0, 1, 1. / self.res) * self.hw,
                             self.ar(0, 1, 1. / self.res) * self.hw)

        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]])

        for i in range(self.n):
            factor = np.exp(- (((I - self.x[i, 0]) ** 2 +
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

    def draw_sprites(self):

        s1 = Sprite(self.x[0, 0]/self.hw, 1-self.x[0, 1]/self.hw, self.shapes[0],
                    c0=255, c1=0, c2=0, scale=self.scale)
        s2 = Sprite(self.x[1, 0]/self.hw, 1-self.x[1, 1]/self.hw, self.shapes[1],
                    c0=0, c1=255, c2=0, scale=self.scale)
        s3 = Sprite(self.x[2, 0]/self.hw, 1-self.x[2, 1]/self.hw, self.shapes[2],
                    c0=0, c1=0, c2=255, scale=self.scale)

        sprites = [s1, s2, s3]
        img = self.renderer.render(sprites)
        return img/255.

    def reset(self):
        """
        Resets the environment to a new configuration
        """
        self.v = np.random.normal(size=(self.n, 2))
        self.v = self.v / norm(self.v) * .5
        self.a = np.zeros_like(self.v)
        self.x = self.init_x()


class BillardsEnv(PhysicsEnv):

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1.,
                 init_v_factor=0, friction_coefficient=0., seed=None, sprites=False):

        super().__init__(n, r, m, hw, granularity, res, t, init_v_factor,
                         friction_coefficient, seed, sprites)

        self.collisions = 0

    def simulate_physics(self, actions=False):
        # F = ma = m dv/dt ---> dv = a * dt = F/m * dt
        v = self.v.copy()

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
                if norm((self.x[i] + self.v[i] * self.t * self.eps) - (self.x[j] + self.v[j] * self.t * self.eps)) < (self.r[i] + self.r[j]):
                    if actions and j == 0:
                        self.collisions = 1
                    w = self.x[i] - self.x[j]
                    w = w / norm(w)

                    v_i = np.dot(w.transpose(), v[i])
                    v_j = np.dot(w.transpose(), v[j])
                    if actions and j == 0:
                        v_j = 0
                    new_v_i, new_v_j = self.new_speeds(self.m[i], self.m[j], v_i, v_j)

                    v[i] += w * (new_v_i - v_i)
                    v[j] += w * (new_v_j - v_j)
                    if actions and j == 0:
                        v[j] = 0

        return v

    def step(self, action, actions=False):
        self.collisions = 0
        return super().step(action, actions=actions)

    def new_speeds(self, m1, m2, v1, v2):
        new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
        new_v1 = new_v2 + (v2 - v1)
        return new_v1, new_v2


class GravityEnv(PhysicsEnv):

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1,
                 init_v_factor=0.18, friction_coefficient=0, seed=None, sprites=False):

        super().__init__(
            n, r, m, hw, granularity, res, t, init_v_factor,
            friction_coefficient, seed, sprites)

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
            x = sc.rand(self.n, 2) * 0.9 * self.hw/2 + self.hw/2
            good_config = True
            for i in range(self.n):
                for j in range(i):
                    good_config = good_config and norm(x[i] - x[j]) > self.hw/3
            counter += 1
        return x

    def init_v(self, factor):
        x_middle = np.sum(self.x, 0) / self.n
        pref = np.random.choice([-1, 1])
        for i in range(self.n):
            v = - (x_middle - self.x[i])
            v = v/norm(v)
            # make noise component wise
            self.v[i] = [pref * v[1] * (factor + 0.13 * sc.randn()),
                        -pref * v[0] * (factor + 0.13 * sc.randn())]

    def step(self, action, actions=False):
        return super().step(action, True, actions=actions)

    def simulate_physics(self, actions=False):
        x_middle = np.array([self.hw/2, self.hw/2])
        v = np.zeros_like(self.v)
        for i in range(self.n):
            F_tot = np.array([0., 0.])
            for j in range(self.n):
                if i != j:
                    r = sc.linalg.norm(self.x[j] - self.x[i])
                    F_tot -= self.G * self.m[j] * self.m[i] * (self.x[i] - self.x[j]) / ((r + 1e-5) ** 3)
            r = (x_middle - self.x[i])
            F_tot += 0.001 * (r ** 3) / norm(r)
            F_tot = np.clip(F_tot, -1, 1)
            v[i] = self.v[i] + (F_tot/self.m[i]) * self.t * self.eps
        return v


class BatchedEnv():

    def __init__(self, env, task, num_envs=16, n=3, r=1., m=1., hw=10,
                 granularity=5, res=32, t=1., init_v_factor=0, friction_coefficient=0.):
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            self.envs.append(task(env(n, r, m, hw, granularity, res, t, init_v_factor, friction_coefficient)))

    def step(self, actions):
        with mp.Pool(processes=self.num_envs) as p:
            results = p.map(self.step_env, [(e, a) for e, a in zip(self.envs, actions)])
        self.envs = [r[1] for r in results]
        results = [r[0] for r in results]
        results = list(zip(*results))
        return results

    def step_env(self, i):
        ret = i[0].step(i[1])
        return ret, i[0]


class ActionPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def next(self):
        raise NotImplementedError("ABC does not implement methods.")


class MonteCarloActionPolicy(ActionPolicy):
    def __init__(self, action_space=9, prob_change=0.1):
        super().__init__(action_space)
        self.p = prob_change
        self.action_arr = range(self.action_space)
        self.current_state = np.random.randint(self.action_space)

    def next(self):
        current_weights = self.p / (self.action_space - 1) * np.ones(self.action_space)
        current_weights[self.current_state] = 1 - self.p
        # assert current_weights.sum() == 1
        rand = np.random.choice(self.action_arr, p=current_weights)

        self.current_state = rand
        return rand


class RandomActionPolicy(ActionPolicy):
    def __init__(self, action_space=9):
        super().__init__(action_space)

    def next(self):
        return np.random.randint(self.action_space)


def main():
    r = 1.
    m = 1.
    env = AvoidanceTask(BillardsEnv(n=3, m=m, r=r, granularity=10, t=1.,
                                    hw=10, res=32),
                        4, greyscale=False)
    T = 100
    all_imgs = np.zeros((T, 32, 32, 3))

    for t in tqdm(range(T)):
        img, state, reward, done = env.step_frame_buffer(np.random.randint(env.get_action_space()))
        if reward < 0:
            img = 1 - img
        all_imgs[t] = img[..., -3:]

    all_imgs = (255 * all_imgs).astype(np.uint8)
    imageio.mimsave('./data/test.gif', all_imgs, fps=24)


def generate_fitting_run(env_class, run_len=100, run_num=1000, max_tries=10000,
                         res=50, n=2, r=1., dt=0.01, gran=10, fc=0.3, hw=10, m=1.,
                         init_v=None, check_overlap=False, sprites=False):

    if init_v is None:
        init_v = [0.1]

    good_counter = 0
    bad_counter = 0
    good_imgs = []
    good_states = []

    for _try in tqdm(range(max_tries)):
        _init_v = np.random.choice(init_v)
        # init_v is ignored for BillardsEnv
        env = env_class(n=n, m=m, granularity=gran, r=r, t=dt, hw=hw, res=res,
                        init_v_factor=_init_v, friction_coefficient=fc,
                        sprites=sprites)
        run_value = 0

        all_imgs = np.zeros((run_len, *env.get_obs_shape()))
        all_states = np.zeros((run_len, env.n, 4))

        run_value = 0
        for t in tqdm(range(run_len)):
            img, state, done = env.step(1)
            all_imgs[t] = img
            all_states[t] = state

            run_value += np.sum(np.logical_and(0 < state[:, :2], state[:, :2] < env.hw)) / (env.n * 2)

            if check_overlap:
                overlap = 0
                for i in range(n):
                    other = list(set(range(n)) - set([i]))
                    # allow small overlaps
                    overlap += np.any(norm(state[i, :2] - state[other, :2]) < 0.9 * (env.r[i] + env.r[other]))
                if overlap > 0:
                    run_value -= 1

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

    print('Generation of {} runs finished, total amount of bad runs: {}. '.format(
        run_num, bad_counter))

    return good_imgs, good_states


def generate_data(save=True, test=False, name='billiards', env=BillardsEnv, config=None):

    num_runs = [1000, 300] if (save and not test) else [2, 5]
    for run_types, run_num in zip(['train', 'test'], num_runs):
        X, y = generate_fitting_run(
            env, run_len=100, run_num=run_num, max_tries=10000, **config)
        data = dict()
        data['X'] = X
        data['y'] = y
        data.update(config)
        data['coord_lim'] = config['hw']
        if save:
            path = './data/{}_{}_data.pkl'.format(name, run_types)
            f = open(path, "wb")
            pickle.dump(data, f, protocol=4)
            f.close()

    first_seq = (255 * X[:20].reshape((-1, config['res'], config['res'], 3))).astype(np.uint8)
    imageio.mimsave('./data/{}.gif'.format(name), first_seq, fps=24)


def generate_billiards_w_actions(
    Task=AvoidanceTask, save=False, config=None, test=False):
    run_len = 100
    action_space = 9
    action_force = 0.6

    num_runs = [1000, 300] if (save and not test) else [2, 10]

    for run_types, run_num in zip(['train', 'test'], num_runs):
        all_imgs = np.zeros((run_num, run_len, config['res'], config['res'], 3))
        all_states = np.zeros((run_num, run_len, config['n'], 4))
        all_actions = np.zeros((run_num, run_len, 9))
        all_rewards = np.zeros((run_num, run_len, 1))
        all_dones = np.zeros((run_num, run_len, 1))

        for run in tqdm(range(run_num)):
            env = Task(BillardsEnv(**config), 4, greyscale=False, action_force=action_force)
            assert action_space == env.get_action_space()
            p = np.random.uniform(0.2, 0.3)
            ap = MonteCarloActionPolicy(action_space=action_space, prob_change=p)
            for t in tqdm(range(run_len)):
                action = ap.next()
                img, state, reward, done = env.step_frame_buffer(action)
                all_imgs[run, t] = img[..., -3:]
                all_states[run, t] = state

                tmp = np.zeros(action_space)
                tmp[action] = 1
                all_actions[run, t-1] = tmp
                all_rewards[run, t] = reward
                all_dones[run, t] = done

        data = dict()
        data['X'] = all_imgs
        data['y'] = all_states
        data['action'] = all_actions
        data['reward'] = all_rewards
        data['done'] = all_dones
        data['type'] = 'max_distance'
        data['action_force'] = action_force

        data.update({'action_space': action_space})
        data.update(config)
        data['coord_lim'] = config['hw']

        if save is not False:
            path = '{}_{}.pkl'.format(save, run_types)
            f = open(path, "wb")
            pickle.dump(data, f, protocol=4)
            f.close()

        first_seq = (255 * all_imgs[:20].reshape((-1, config['res'], config['res'], 3))).astype(np.uint8)
        if save is False:
            save = 'action'
        imageio.mimsave('{}.gif'.format(save), first_seq, fps=24)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()

    config = {'res': 32, 'hw': 10, 'n': 3, 'dt': 1, 'm': 1., 'fc': 0,
              'gran': 2, 'r': 1.2, 'check_overlap': False}
    generate_data(
        test=args.test, name='billiards', env=BillardsEnv, config=config)

    # config.update({'sprites': True})
    # generate_data(
    #     test=args.test, name='billards_sprites', env=BillardsEnv, config=config)

    config = {'res': 50, 'hw': 30, 'n': 3, 'dt': 1, 'm': 4., 'fc': 0,
              'init_v': [0.55], 'gran': 50, 'r': 2, 'check_overlap': True}

    generate_data(
        test=args.test, name='gravity', env=GravityEnv, config=config)

    # config.update({'sprites': True})
    # generate_data(
    #     test=args.test, name='gravity_sprites', env=GravityEnv, config=config)

    config = {
        'res': 32, 'hw': 10, 'n': 3, 't': 1., 'm': 1.,
        'granularity': 50, 'r': 1, 'friction_coefficient': 0}

    generate_billiards_w_actions(
        config=config, save='./data/avoidance', test=args.test)
