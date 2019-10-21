"""Contains code for data set creation as well as live environments."""

import argparse
import pickle
import imageio
import numpy as np
import scipy as sc
import multiprocessing as mp
from tqdm import tqdm

from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite


def norm(x):
    """Overloading numpys default behaviour for norm()."""
    if len(x.shape) == 1:
        _norm = np.linalg.norm(x)
    else:
        _norm = np.linalg.norm(x, axis=1).reshape(-1, 1)
    return _norm


class Task():
    """Defines a task for interactive environments.

    For all tasks defined here, actions correspond to direct movements of the
    controlled balls. Rewards are defined by the derived classes.
    """

    angular = 1. / np.sqrt(2)
    action_selection = [
        np.array([0., 0.]),
        np.array([1., 0.]),
        np.array([0., 1.]),
        np.array([angular, angular]),
        np.array([-1., 0.]),
        np.array([0., -1.]),
        np.array([-angular, -angular]),
        np.array([-angular, angular]),
        np.array([angular, -angular])]

    def __init__(self, env, num_stacked=4, greyscale=False, action_force=.3):
        """Initialise task.

        Args:
            env (Environment): Tasks have environments as attribute.
            num_stacked (int): Create a frame buffer of num_stacked images.
            greyscale (bool): Convert rgb images to 'greyscale'.
            action_force (float): Distance moved per applied action.
        """
        self.env = env

        # make controlled ball quasi-static
        self.env.m[0] = 10000

        if greyscale:
            self.frame_buffer = np.zeros(
                (*env.get_obs_shape()[:2], num_stacked))
            self.conversion = lambda x: np.sum(
                x * [[[0.3, 0.59, 0.11]]], 2, keepdims=True)
        else:
            sh = env.get_obs_shape()
            self.frame_buffer = np.zeros((*sh[:2], sh[2] * num_stacked))
            self.conversion = lambda x: x

        self.frame_channels = 3 if not greyscale else 1
        self.action_force = action_force

    def get_action_space(self):
        """Return number of available actions."""
        return len(self.action_selection)

    def get_framebuffer_shape(self):
        """Return shape of frame buffer."""
        return self.frame_buffer.shape

    def calculate_reward(self, state, action, env):
        """Abstract method. To be overwritten by derived classes."""
        raise NotImplementedError

    def resolve_action(self, _action, env=None):
        """Implement the effects of an action. Change this to change action."""
        action = self.action_selection[_action]
        action = action * self.action_force
        return action

    def step(self, _action):
        """Propagate env to next step."""
        action = self.resolve_action(_action)
        img, state, done = self.env.step(action)
        r = self.calculate_reward()
        return img, state, r, done

    def step_frame_buffer(self, _action=None):
        """Step environment with frame buffer."""
        action = self.resolve_action(_action)
        img, state, done = self.env.step(action)
        r = self.calculate_reward()

        img = self.conversion(img)
        c = self.frame_channels
        self.frame_buffer[:, :, :-c] = self.frame_buffer[:, :, c:]
        self.frame_buffer[:, :, -c:] = img

        return self.frame_buffer, state, r, done


class AvoidanceTask(Task):
    """Derived Task: Avoidance Task."""

    def calculate_reward(self,):
        """Negative sparse reward of -1 is given in case of collisions."""
        return -self.env.collisions

class MaxDistanceTask(Task):
    """Derived Task: Maximal Distance Task."""

    def calculate_reward(self):
        """Continuous reward is given.

        Negative reward is given in dependence of the minimal distance of the
        controlled ball to any other ball.
        """
        scaling = 2
        r = 0
        for i in range(1, self.env.n):
            current_norm = norm(self.env.x[i, 0:2] - self.env.x[0, 0:2])\
                - 2 * self.env.r[0]
            current_exp = -np.clip(np.exp(-current_norm * scaling), 0, 1)
            r = min(r, current_exp)
        return r


class MinDistanceTask(Task):
    """Derived Task: Minimal Distance Task."""

    def calculate_reward(self, state, action, env):
        """Continuous reward is given.

        Controlled ball is incentivised to follow any of the other balls.
        Reward is always negative, unless the controlled ball touches any of the
        other balls. Negative reward is given for the distance to the nearest
        ball to the controlled ball.
        """
        # initialize r to very small reward (~ -inf)
        r = - ((100 * env.hw) ** 2)
        for i in range(1, env.n):
            r = max(r,
                    -(norm(state[i, 0:2] - state[0, 0:2]) - 2 * env.r[0]) ** 2)
        return r


class PhysicsEnv:
    """Base class for the physics environments."""

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1.,
                 init_v_factor=0, friction_coefficient=0., seed=None,
                 sprites=False):
        """Initialize a physics env with some general parameters.

        Args:
            n (int): Optional, number of objects in the scene.
            r (float)/list(float): Optional, radius of objects in the scene.
            m (float)/list(float): Optional, mass of the objects in the scene.
            hw (float): Optional, coordinate limits of the environment.
            eps (float): Optional, internal simulation granularity as the
                fraction of one time step. Does not change speed of simulation.
            res (int): Optional, pixel resolution of the images.
            t (float): Optional, dt of the step() method. Speeds up or slows
                down the simulation.
            init_v_factor (float): Scaling factor for inital velocity. Used only
                in Gravity Environment.
            friction_coefficient (float): Friction slows down balls.
            seed (int): Set random seed for reproducibility.
            sprites (bool): Render selection of sprites using spriteworld
                instead of balls.

        """
        np.random.seed(seed)

        self.n = n
        self.r = np.array([[r]] * n) if np.isscalar(r) else r
        self.m = np.array([[m]] * n) if np.isscalar(m) else m
        self.hw = hw
        self.internal_steps = granularity
        self.eps = 1 / granularity
        self.res = res
        self.t = t

        self.x = self.init_x()
        self.v = self.init_v(init_v_factor)
        self.a = np.zeros_like(self.v)

        self.fric_coeff = friction_coefficient
        self.v_rotation_angle = 2 * np.pi * 0.05

        if n > 3:
            self.use_colors = True
        else:
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
            self.scale = self.r[0] / self.hw / 0.6
            self.shapes = np.random.choice(shapes, 3)
            self.draw_image = self.draw_sprites

        else:
            self.draw_image = self.draw_balls

    def init_v(self, init_v_factor):
        """Randomly initialise velocities."""
        v = np.random.normal(size=(self.n, 2))
        v = v / np.sqrt((v ** 2).sum()) * .5
        return v

    def init_x(self):
        """Initialize ojbject positions without overlap and in bounds."""
        good_config = False
        while not good_config:
            x = np.random.rand(self.n, 2) * self.hw / 2 + self.hw / 4
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

    def simulate_physics(self, actions):
        """Calculates physics for a single time step.

        What "physics" means is defined by the respective derived classes.

        Args:
            action (np.Array(float)): A 2D-float giving an x,y force to
                enact upon the first object.

        Returns:
            d_vs (np.Array(float)): Velocity updates for the simulation.

        """
        raise NotImplementedError

    def step(self, action=None, mass_center_obs=False):
        """Full step for the environment."""
        if action is not None:
            # Actions are implemented as hardly affecting the first object's v.
            self.v[0] = action * self.t
            actions = True

        else:
            actions = False

        for _ in range(self.internal_steps):
            self.x += self.t * self.eps * self.v

            if mass_center_obs:
                # Do simulation in center of mass system.
                c_body = np.sum(self.m * self.x, 0) / np.sum(self.m)
                self.x += self.hw / 2 - c_body

            self.v -= self.fric_coeff * self.m * self.v * self.t * self.eps
            self.v = self.simulate_physics(actions)

        img = self.draw_image()
        state = np.concatenate([self.x, self.v], axis=1)
        done = False

        return img, state, done

    def get_obs_shape(self):
        """Return image dimensions."""
        return (self.res, self.res, 3)

    def get_state_shape(self):
        """Get shape of state array."""
        state = np.concatenate([self.x, self.v], axis=1)
        return state.shape

    @staticmethod
    def ar(x, y, z):
        """Offset array function."""
        return z / 2 + np.arange(x, y, z, dtype='float')

    def draw_balls(self):
        """Render balls on canvas."""
        if self.n > 3 and not self.use_colors:
            raise ValueError(
                'Must self.use_colors if self.n > 3.')

        if self.n > 6:
            raise ValueError(
                'Max self.n implemented currently is 6.')

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
        """Render sprites on the current locations."""

        s1 = Sprite(self.x[0, 0] / self.hw, 1 - self.x[0, 1] / self.hw,
                    self.shapes[0],
                    c0=255, c1=0, c2=0, scale=self.scale)
        s2 = Sprite(self.x[1, 0] / self.hw, 1 - self.x[1, 1] / self.hw,
                    self.shapes[1],
                    c0=0, c1=255, c2=0, scale=self.scale)
        s3 = Sprite(self.x[2, 0] / self.hw, 1 - self.x[2, 1] / self.hw,
                    self.shapes[2],
                    c0=0, c1=0, c2=255, scale=self.scale)

        sprites = [s1, s2, s3]
        img = self.renderer.render(sprites)

        return img / 255.

    def reset(self, init_v_factor=None):
        """Resets the environment to a new configuration."""
        self.v = self.init_v(init_v_factor)
        self.a = np.zeros_like(self.v)
        self.x = self.init_x()


class BillardsEnv(PhysicsEnv):
    """Billiards or Bouncing Balls environment."""

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1.,
                 init_v_factor=0, friction_coefficient=0., seed=None, sprites=False):
        """Initialise arguments of parent class."""
        super().__init__(n, r, m, hw, granularity, res, t, init_v_factor,
                         friction_coefficient, seed, sprites)

        # collisions is updated in step to measure the collisions of the balls
        self.collisions = 0

    def simulate_physics(self, actions):
        # F = ma = m dv/dt ---> dv = a * dt = F/m * dt
        v = self.v.copy()

        # check for collisions with wall
        for i in range(self.n):
            for z in range(2):
                next_pos = self.x[i, z] + (v[i, z] * self.eps * self.t)
                # collision at 0 wall
                if not self.r[i] < next_pos:
                    self.x[i, z] = self.r[i]
                    v[i, z] = - v[i, z]
                # collision at hw wall
                elif not next_pos < (self.hw - self.r[i]):
                    self.x[i, z] = self.hw - self.r[i]
                    v[i, z] = - v[i, z]

        # check for collisions with objects
        for i in range(self.n):
            for j in range(i):

                dist = norm((self.x[i] + v[i] * self.t * self.eps)
                            - (self.x[j] + v[j] * self.t * self.eps))

                if dist < (self.r[i] + self.r[j]):
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

    def new_speeds(self, m1, m2, v1, v2):
        """Implement elastic collision between two objects."""
        new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
        new_v1 = new_v2 + (v2 - v1)
        return new_v1, new_v2

    def step(self, action=None):
        """Overwrite step functino to ensure collisions are zeroed beforehand."""
        self.collisions = 0
        return super().step(action)


class GravityEnv(PhysicsEnv):
    """Derived Task: Minimal Distance Task."""

    def __init__(self, n=3, r=1., m=1., hw=10, granularity=5, res=32, t=1,
                 init_v_factor=0.18, friction_coefficient=0, seed=None,
                 sprites=False):
        """Initialise arguments of parent class."""

        super().__init__(
            n, r, m, hw, granularity, res, t, init_v_factor,
            friction_coefficient, seed, sprites)

        self.G = 0.5
        self.K1 = self.G
        self.K2 = 1

    def init_x(self):
        """Initialize object positions without overlap and in bounds.

        To achieve a stable gravity configuration, default init is overwritten.
        Here, objects are initialised with more padding.
        """
        good_config = False
        counter = 0
        while not good_config and counter < 1000:
            x = sc.rand(self.n, 2) * 0.9 * self.hw / 2 + self.hw / 2
            good_config = True
            for i in range(self.n):
                for j in range(i):
                    good_config = good_config and norm(
                        x[i] - x[j]) > self.hw / 3
            counter += 1
        return x

    def init_v(self, factor):
        """Initialize a stable velocity configuration.

        Velocities are initialised as orthogonal to the object's position vector
        as measured from the center.
        """
        x_middle = np.sum(self.x, 0) / self.n
        pref = np.random.choice([-1, 1])
        full_v = np.zeros((self.n, 2))

        for i in range(self.n):
            v = - (x_middle - self.x[i])
            v = v / norm(v)
            # make noise component wise
            full_v[i] = np.array([pref * v[1] * (factor + 0.13 * sc.randn()),
                            -pref * v[0] * (factor + 0.13 * sc.randn())])
        return full_v

    def step(self, action=None):
        """Set actions to false by default."""
        return super().step(action, True)

    def simulate_physics(self, actions):
        """Simulate gravitational physics.

        Additional attractive force towards the center is applied for stability.
        Forces are clipped to avoid slingshotting effects.
        """
        x_middle = np.array([self.hw/2, self.hw/2])
        v = np.zeros_like(self.v)

        for i in range(self.n):
            F_tot = np.array([0., 0.])

            for j in range(self.n):
                if i != j:
                    r = np.linalg.norm(self.x[j] - self.x[i])
                    F_tot -= self.G * self.m[j] * self.m[i] * (
                                self.x[i] - self.x[j]) / ((r + 1e-5) ** 3)

            r = (x_middle - self.x[i])
            F_tot += 0.001 * (r ** 3) / norm(r)
            F_tot = np.clip(F_tot, -1, 1)
            v[i] = self.v[i] + (F_tot / self.m[i]) * self.t * self.eps

        return v


class ActionPolicy:
    """Abstract base class for action policy.

    An action policy specifies a series of actions.
    """

    def __init__(self, action_space):
        """Initialise action policy.

        Args:
            action_space (int): Number of available actions.
        """
        self.action_space = action_space

    def next(self):
        raise NotImplementedError("ABC does not implement methods.")


class RandomActionPolicy(ActionPolicy):
    """Random action policy."""

    def __init__(self, action_space=9):
        """Initialise random action policy."""
        super().__init__(action_space)

    def next(self):
        """Next action is given completely independent of history."""
        return np.random.randint(self.action_space)


class MonteCarloActionPolicy(ActionPolicy):
    """Monte carlo action policy.

    First action is chosen randomly. After, action is only changed with
    prob_change probability.
    """

    def __init__(self, action_space=9, prob_change=0.1):
        """Initialise monte carlo action policy.

        Args:
            prob_change (float): Probability of changing action from t to t+1.

        """
        super().__init__(action_space)
        self.p = prob_change
        self.action_arr = range(self.action_space)
        self.current_state = np.random.randint(self.action_space)

    def next(self):
        """Get next action given current."""
        action_space = self.action_space
        current_weights = self.p / (action_space - 1) * np.ones(action_space)
        current_weights[self.current_state] = 1 - self.p
        # assert current_weights.sum() == 1

        self.current_state = np.random.choice(self.action_arr,
                                              p=current_weights)
        return self.current_state


def generate_fitting_run(env_class, run_len=100, run_num=1000, max_tries=10000,
                         res=50, n=2, r=1., dt=0.01, gran=10, fc=0.3, hw=10,
                         m=1., seed=None,
                         init_v=None, check_overlap=False, sprites=False):
    """Generate runs for environments.

    Integrated error checks. Parameters as passed to environments.
    """

    if init_v is None:
        init_v = [0.1]

    good_counter = 0
    bad_counter = 0
    good_imgs = []
    good_states = []

    for _try in tqdm(range(max_tries)):
        _init_v = np.random.choice(init_v)
        # init_v is ignored for BillardsEnv
        env = env_class(n=n, r=r, m=m, hw=hw, granularity=gran, res=res, t=dt,
                        init_v_factor=_init_v, friction_coefficient=fc, seed=seed,
                        sprites=sprites)
        run_value = 0

        all_imgs = np.zeros((run_len, *env.get_obs_shape()))
        all_states = np.zeros((run_len, env.n, 4))

        run_value = 0
        for t in tqdm(range(run_len)):

            img, state, _ = env.step()
            all_imgs[t] = img
            all_states[t] = state

            run_value += np.sum(np.logical_and(
                state[:, :2] > 0, state[:, :2] < env.hw)) / (env.n * 2)

            if check_overlap:

                overlap = 0
                for i in range(n):
                    other = list(set(range(n)) - {i, })
                    # allow small overlaps
                    overlap += np.any(norm(state[i, :2] - state[other, :2])
                                      < 0.9 * (env.r[i] + env.r[other]))

                if overlap > 0:
                    run_value -= 1

        if run_value > (run_len - run_len / 100):
            good_imgs.append(all_imgs)
            good_states.append(all_states)
            good_counter += 1
        else:
            bad_counter += 1

        if good_counter >= run_num:
            break

    good_imgs = np.stack(good_imgs, 0)
    good_states = np.stack(good_states, 0)

    print(
        'Generation of {} runs finished, total amount of bad runs: {}. '.format(
            run_num, bad_counter))

    return good_imgs, good_states


def generate_data(save=True, test_gen=False, name='billiards', env=BillardsEnv,
                  config=None):
    """Generate data for billiards or gravity environment."""

    num_runs = [1000, 300] if (save and not test_gen) else [2, 5]
    for run_types, run_num in zip(['train', 'test'], num_runs):

        # generate runs
        X, y = generate_fitting_run(
            env, run_len=100, run_num=run_num, max_tries=10000, **config)

        # save data
        data = dict()
        data['X'] = X
        data['y'] = y
        data.update(config)
        data['coord_lim'] = config['hw']

        if save:
            path = './data/{}_{}.pkl'.format(name, run_types)
            f = open(path, "wb")
            pickle.dump(data, f, protocol=4)
            f.close()

    # also generate gif of data
    first_seq = (255 * X[:20].reshape(
        (-1, config['res'], config['res'], 3))).astype(np.uint8)
    imageio.mimsave('./data/{}.gif'.format(name), first_seq, fps=24)


def generate_billiards_w_actions(ChosenTask=AvoidanceTask, save=True,
                                 config=None, test_gen=False):
    run_len = 100
    action_space = 9
    action_force = 0.6

    num_runs = [1000, 300] if (save and not test_gen) else [2, 10]

    for run_types, run_num in zip(['train', 'test'], num_runs):

        all_imgs = np.zeros(
            (run_num, run_len, config['res'], config['res'], 3))
        all_states = np.zeros((run_num, run_len, config['n'], 4))
        all_actions = np.zeros((run_num, run_len, 9))
        all_rewards = np.zeros((run_num, run_len, 1))
        all_dones = np.zeros((run_num, run_len, 1))

        # number of sequences
        for run in tqdm(range(run_num)):
            env = ChosenTask(BillardsEnv(**config),
                             4, greyscale=False, action_force=action_force)

            assert action_space == env.get_action_space()
            p = np.random.uniform(0.2, 0.3)
            ap = MonteCarloActionPolicy(action_space=action_space,
                                        prob_change=p)

            # number of steps per sequence
            for t in tqdm(range(run_len)):
                action = ap.next()
                img, state, reward, done = env.step(action)
                all_imgs[run, t] = img
                all_states[run, t] = state

                tmp = np.zeros(action_space)
                tmp[action] = 1
                all_actions[run, t - 1] = tmp
                all_rewards[run, t] = reward
                all_dones[run, t] = done

        # save results
        data = dict()
        data['X'] = all_imgs
        data['y'] = all_states
        data['action'] = all_actions
        data['reward'] = all_rewards
        data['done'] = all_dones
        # still a bit hacky, need to implement __str__
        if ChosenTask is not AvoidanceTask:
            raise ValueError
        data['type'] = 'AvoidanceTask'
        data['action_force'] = action_force

        data.update({'action_space': action_space})
        data.update(config)
        data['coord_lim'] = config['hw']

        if save:
            path = 'data/avoidance_{}.pkl'.format(run_types)
            f = open(path, "wb")
            pickle.dump(data, f, protocol=4)
            f.close()

        # example sequences as gif
        res = config['res']
        first_seq = (255 * all_imgs[:20].reshape((-1, res, res, 3)))
        first_seq = first_seq.astype(np.uint8)
        imageio.mimsave('data/avoidance.gif'.format(save), first_seq, fps=24)


def main(script_args):
    """Create standard collection of data sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-gen', dest='test_gen', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    args = parser.parse_args(script_args)

    config = {'res': 32, 'hw': 10, 'n': 3, 'dt': 1, 'm': 1., 'fc': 0,
              'gran': 2, 'r': 1.2, 'check_overlap': False}

    generate_data(
        save=args.save, test_gen=args.test_gen, name='billiards',
        env=BillardsEnv, config=config)

    # config.update({'sprites': True})
    # generate_data(
    #     test_gen=args.test_gen, name='billards_sprites', env=BillardsEnv, config=config)

    config = {'res': 50, 'hw': 30, 'n': 3, 'dt': 1, 'm': 4., 'fc': 0,
              'init_v': [0.55], 'gran': 50, 'r': 2, 'check_overlap': True}

    generate_data(
        save=args.save, test_gen=args.test_gen, name='gravity',
        env=GravityEnv, config=config)

    # config.update({'sprites': True})
    # generate_data(
    #     test_gen=args.test_gen, name='gravity_sprites', env=GravityEnv, config=config)

    config = {
        'res': 32, 'hw': 10, 'n': 3, 't': 1., 'm': 1.,
        'granularity': 50, 'r': 1, 'friction_coefficient': 0}

    generate_billiards_w_actions(
        config=config, save=args.save, test_gen=args.test_gen)
