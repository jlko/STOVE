import pygame
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import zoom
import torch

import envs
from vin import main as load_vin


def encode_img(img):
    img_torch = torch.Tensor(img)
    img_torch = img_torch.permute(0, 1, 4, 2, 3)
    # infer initial model state
    return img_torch


def load_model():
    restore = './good_rl_run'
    extras = {'nolog':True,
              'traindata': './rl_data/billards_w_actions_train_data.pkl',
              'testdata': './rl_data/billards_w_actions_test_data.pkl'}
    trainer = load_vin(extras=extras, restore=restore)
    model = trainer.net
    return model


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def init_model(model):
    # get 8 images from a real env to initialize model
    # use batch_size 2 to avoid getting squeezed!
    real_env = make_env()
    initial_imgs = np.zeros((2, 8, 32, 32, 3))
    initial_actions = np.zeros((2, 8, 9))
    initial_actions[:, :, 3] = 1.
    initial_actions = torch.tensor(initial_actions)

    for i in range(8):
        ret_img, _, _, _ = real_env.step(3)
        initial_imgs[:, i] = ret_img
    _, init_state, _ = model(encode_img(initial_imgs), 0, action=initial_actions,
                             pretrain=False, future=None)
    return init_state['z'], init_state['obj_appearances']


def get_action_from_presses(pressed):
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
    config = {'res': 32, 'hw': 15, 'n': 3, 't': 1, 'm': 1.,  #dt = 0.81
              'granularity': 10, 'r': 1}
    return envs.MaxDistanceTask(envs.BillardsEnv(**config),
            4, greyscale=False, action_force=0.6)

def draw_result_screen(display, font, score, cur_game, total_games):
    score_text = font.render('Score {:.2f}'.format(float(score)), True, (255, 255, 255), (0, 0, 0))
    games_text = font.render("{}/{} games played".format(cur_game, total_games),
                             True, (255, 255, 255), (0, 0, 0))
    press_space_text = font.render("Press SPACE to cont.", True, (255, 255, 255), (0, 0, 0))
    display.blit(score_text, (00, 20))
    display.blit(games_text, (00, 60))
    display.blit(press_space_text, (00, 100))

def play_game(use_model=False):
    if use_model:
        model = load_model()
        cur_state, appearances = init_model(model)
        # appearances = tile(appearances, 0, 9)
        appearances = appearances[:, -1]
        cur_state = cur_state[:, -1]

    pygame.init()
    display = pygame.display.set_mode((128, 148))
    font = pygame.font.Font('freesansbold.ttf', 11)
    env = make_env()

    action_space = 9
    assert action_space == env.get_action_space()

    state = 'play'
    clock = pygame.time.Clock()
    score = 0.
    games_played = 0
    total_games = 50
    frame_num = 0
    max_frames = 300
    while state != 'quit':
        clock.tick(50)
        print(clock.get_time())
        display.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state = 'quit'

        pressed = pygame.key.get_pressed()
        if state == 'play':
            action = get_action_from_presses(pressed)
            if use_model:
                cur_actions = np.zeros((2, 1, 9))
                cur_actions[:, 0, action] = 1.
                cur_actions = torch.tensor(cur_actions)
                cur_state, reward = model.rollout(cur_state, 1, actions=cur_actions,
                                                  appearance=appearances)
                img = model.reconstruct_from_z(cur_state)
                cur_state = cur_state[:, 0]
                img = img[0, 0, 0].unsqueeze(-1)
                reward = reward[0].item() - 1
            else:
                img, _, reward, done = env.step(action)
            frame_num += 1
            score += reward
            text = font.render('{:.2f}'.format(float(score)), True, (255, 255, 255), (0, 0, 0))
            # img_resize = imresize(img, (64, 64), 'nearest')
            channel_extension = 3. if use_model else 1.
            img_resize = zoom(img, (4., 4., channel_extension))
            img_resize = np.clip(img_resize, 0., 1.)
            img_arr = pygame.surfarray.make_surface(img_resize * 255)
            display.blit(text, (0, 0))
            display.blit(img_arr, (0, 20))
            if frame_num == max_frames:
                games_played += 1
                if games_played == max_frames:
                    state = 'quit'
                else:
                    state = 'score'
        elif state == 'score':
            draw_result_screen(display, font, score, games_played, total_games)
            if pressed[pygame.K_SPACE]:
                state = 'play'
                score = 0.
                frame_num = 0
                if use_model:
                    model = load_model()
                    cur_state, appearances = init_model(model)
                    # appearances = tile(appearances, 0, 9)
                    appearances = appearances[:, -1]
                    cur_state = cur_state[:, -1]
                else:
                    env = make_env()

        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    with torch.no_grad():
        play_game(True)
