"""Play the Billiards RL environment as interactive game.

Also allows to play inside the STOVE world-model.
"""
import os
import argparse
import pygame
import numpy as np
from scipy.ndimage import zoom
import torch

from model.envs import envs
from model.main import restore_model

def encode_img(img):
    """Format image."""
    img_torch = torch.Tensor(img)
    img_torch = img_torch.permute(0, 1, 4, 2, 3)
    return img_torch


def tile(a, dim, n_tile):
    """Tiles appearances."""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([
            init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def init_model(model):
    """For STOVE as world model: Initialise env for STOVE generation.

    Args:
        model (STOVE): Initialised model.

    Get 8 images from a real env to initialize STOVE.
    Use batch_size 2 to avoid getting squeezed.

    """
    real_env = make_env()
    initial_imgs = np.zeros((2, 8, 32, 32, 3))
    initial_actions = np.zeros((2, 8, 9))
    initial_actions[:, :, 3] = 1.
    initial_actions = torch.tensor(initial_actions)

    for i in range(8):
        ret_img, _, _, _ = real_env.step(3)
        initial_imgs[:, i] = ret_img

    _, init_state, _ = model(
        encode_img(initial_imgs), 0, actions=initial_actions,)

    return init_state['z'], init_state['obj_appearances']


def get_action_from_presses(pressed):
    """Transform button presses to env actions."""
    up = pressed[pygame.K_UP] or pressed[pygame.K_w]
    down = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
    left = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
    right = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]

    if down and right:
        return 3
    if down and left:
        return 8
    if up and right:
        return 7
    if up and left:
        return 6
    if up:
        return 4
    if down:
        return 1
    if right:
        return 2
    if left:
        return 5
    return 0


def make_env():
    """Create environment."""
    config = {
        'res': 32, 'hw': 10, 'n': 3, 't': 1., 'm': 1.,
        'granularity': 50, 'r': 1, 'friction_coefficient': 0}

    return envs.AvoidanceTask(
        envs.BillardsEnv(**config), 4, action_force=0.6)


def draw_result_screen(display, font, mean_rew, color, tw, th):
    """Draw result screen."""
    white = (255, 255, 255)
    black = (0, 0, 0)

    score_str1 = 'Mean Reward {:.2f}'.format(mean_rew)
    score_text1 = font.render(score_str1, True, color, black)

    if mean_rew > -1:
        judgement_str = 'Better than AI!'
    else:
        judgement_str = 'Worse than AI!'

    judgement_text = font.render(judgement_str, True, color, black)

    press_space_str = 'Continue with SPACE!'
    press_space_text = font.render(press_space_str, True, color, black)

    display.blit(score_text1, (tw, th+20))
    display.blit(judgement_text, (tw, th+120))
    display.blit(press_space_text, (tw, th+220))


def play_game(use_model=None, no_reset=True, no_blinky=False):
    """Execute main game loop."""
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    if use_model is not None:
        model = restore_model(use_model)
        cur_state, appearances = init_model(model)
        # appearances = tile(appearances, 0, 9)
        appearances = appearances[:, -1]
        cur_state = cur_state[:, -1]

    env_res = 32
    zoom_factor = 4
    res = zoom_factor*4 * env_res
    top = 40

    pygame.init()

    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h

    top_width = screen_width//2 - res//2
    top_height = screen_height//2 - res//2
    tw, th = top_width, top_height

    # display = pygame.display.set_mode((res, res+top), pygame.FULLSCREEN)
    display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    font = pygame.font.Font('freesansbold.ttf', 40)
    font2 = pygame.font.Font('freesansbold.ttf', 80)

    env = make_env()

    action_space = 9
    assert action_space == env.get_action_space()

    state = 'play'
    clock = pygame.time.Clock()
    rewards = []
    frame_num = 0
    max_frames = 100 * 10
    if not no_blinky:
        bg_color = (0, 255, 0)
    else:
        bg_color = (0, 0, 0)

    while state != 'quit':
        clock.tick(50)

        display.fill(bg_color)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state = 'quit'

        pressed = pygame.key.get_pressed()
        if state == 'play':
            action = get_action_from_presses(pressed)

            if use_model is not None:
                cur_actions = np.zeros((2, 1, 9))
                cur_actions[:, 0, action] = 1.
                cur_actions = torch.tensor(cur_actions)
                cur_state, reward = model.rollout(
                    cur_state, 1, actions=cur_actions, appearance=appearances)
                img = model.reconstruct_from_z(cur_state)
                cur_state = cur_state[:, 0]
                img = img[0, 0, 0].unsqueeze(-1)
                reward = reward[0].item() - 1
            else:
                img, _, reward, _ = env.step(action)

            if reward == -1:
                img = 1-img

            frame_num += 1

            if not no_reset and (frame_num % 100 == 0):
                env.env.reset()

            rewards.append(reward)
            mean_rew = np.mean(rewards) * 100

            if mean_rew > -1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            if not no_blinky:
                bg_color = color

            info_str = 'Mean Reward: {:.2f}'

            text = font.render(
                info_str.format(mean_rew), True, color, (0, 0, 0))

            channel_extension = 3. if use_model else 1.
            img_resize = zoom(img, (4, 4, channel_extension))
            img_resize = np.clip(img_resize, 0., 1.)
            img_resize[0, :] = 1
            img_resize[-1, :] = 1
            img_resize[:, 0] = 1
            img_resize[:, -1] = 1
            img_arr = pygame.surfarray.make_surface(img_resize * 255)
            img_arr = pygame.transform.scale(img_arr, (res, res))

            display.blit(text, (tw, th))
            display.blit(img_arr, (tw, th+top))

            if (frame_num == max_frames) or pressed[pygame.K_ESCAPE]:
                state = 'score'

        elif state == 'score':
            draw_result_screen(display, font2, mean_rew, color, tw, th)
            if pressed[pygame.K_SPACE]:
                state = 'play'
                rewards = []
                frame_num = 0
                if use_model is not None:
                    model = load_model(use_model)
                    cur_state, appearances = init_model(model)
                    # appearances = tile(appearances, 0, 9)
                    appearances = appearances[:, -1]
                    cur_state = cur_state[:, -1]
                else:
                    env = make_env()
        if pressed[pygame.K_c]:
            state = 'quit'

        pygame.display.update()

    pygame.quit()


def main(script_args):
    """Parse args and run game."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default=None,
                        help='If specified, play happens in STOVE world model.')
    parser.add_argument('--no-reset', dest='no_reset', action='store_true')
    parser.add_argument('--no-blinky', dest='no_blinky', action='store_true')

    args = parser.parse_args(script_args)
    with torch.no_grad():
        play_game(args.path, args.no_reset, args.no_blinky)
