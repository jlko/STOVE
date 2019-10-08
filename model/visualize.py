import os
import matplotlib.pyplot as plt
import numpy as np
import imageio

# VIN Plots
color = ['r', 'g', 'b', 'k', 'y', 'm', 'c']


def ar(x, y, z):
    return z / 2 + np.arange(x, y, z, dtype='float')

def draw_image(X, res, r=None, size=None):
    if size is None:
        size = 10

    T, n = X.shape[0:2]
    if r is None:
        r = np.array([1.2] * n)

    A = np.zeros((T, res, res, 3), dtype='float')

    [I, J] = np.meshgrid(ar(0, 1, 1. / res) * size, ar(0, 1, 1. / res) * size)

    # time
    for t in range(T):
        # objects
        for i in range(n):
            # this is smoothing kernel for ball intensity.
            # i.e. the further away we are from ball center, the lower
            # the intensity is.
            A[t, :, :, i] += np.exp(-(((I - X[t, i, 0]) ** 2 +
                                    (J - X[t, i, 1]) ** 2) /
                                   (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A

def plot_positions(xy, img_folder, prefix, save=True, size=10, presentation=False):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    from visualize import color
    fig_num = len(xy)
    mydpi = 100
    fig = plt.figure(figsize=(128 * 3/mydpi, 128 * 3/mydpi))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    if presentation:
        color = 10*['k']
        transparent = True
    else:

        transparent = False

    for i in range(fig_num):
        for j in range(len(xy[0])):
            # make y start at top left
            plt.scatter(xy[i, j, 1], 10 - xy[i, j, 0],
                        c=color[j % len(color)], s=size, alpha=(i+1)/fig_num)

    plt.tight_layout()

    if save:
        fig.savefig(img_folder+prefix+".pdf", dpi=mydpi, transparent=transparent)
        vis.matplot(fig)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def animate_mpl(states, img_folder, prefix, presentation=False):
    from PIL import Image
    images = []
    for i in range(len(states)):
        img = plot_positions(states[i:i + 1], img_folder,
                             prefix, save=False, size=270*12, presentation=presentation)
        images.append(img)
    imageio.mimsave(os.path.join(img_folder, prefix+'.gif'), images, fps=24)


def animate(states, img_folder, prefix, res=32, r=1.2, size=None, rewards=None):
    '''param states: shape (T, o, 2) sequence to animate'''

    if img_folder is not None and not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # length of sequence
    num_img, num_obj, _ = states.shape
    # isolate positions
    xy = states[..., :2]
    # radii of ball, should be identical to
    # how data was created
    _r = np.array(3 * [r])
    images = draw_image(xy, res=res, r=_r, size=size)

    images = (255 * images).astype('uint8')

    if rewards is not None:
        images[rewards < 0.5] = 255 - images[rewards < 0.5]
        reward_annotations = get_reward_annotation(rewards, res=res)
        reward_annotations = reward_annotations.astype('uint8')
        images = np.concatenate([reward_annotations, images], 1)
        images = images.astype('uint8')

    if img_folder is not None:
        imageio.mimsave(os.path.join(img_folder, prefix+'.gif'), images, fps=24)
    return images

from PIL import Image, ImageDraw
def get_reward_annotation(rewards, res):
    h = 15
    annotations = np.zeros((rewards.size, h, res, 3))
    for t, reward in enumerate(rewards):
        img = Image.new('RGB', (res, h), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((0, 0), '{:.2f}'.format(reward), fill=(255, 255, 0),)
        annotations[t] = np.array(img)
    return annotations


# Object Detection Plots

# from crazyK, super useful
def setup_axis(axis):
    axis.tick_params(axis='both',       # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # ticks along the bottom edge are off
                     top=False,         # ticks along the top edge are off
                     right=False,
                     left=False,
                     labelbottom=False,
                     labelleft=False)   # labels along the bottom edge are off


def rectangles_from_z(z, ax, width, height):
    cs = ['white', 'yellow', 'orange']
    for c, obj in zip(cs, z):
        sy, sx, y, x = obj

        # imshow has 0,0 top left, for our transf its in center
        x += 1
        y += 1

        # x,y need to go from [0, 2] to [0, 32]
        x *= width/2
        y *= height/2
        sx *= width
        sy *= height

        # corners of bounding box
        left_x = x - 0.5 * sx
        right_x = x + 0.5 * sx
        high_y = y - 0.5 * sy
        low_y = y + 0.5 * sy

        # plot bounding box
        ax.plot([left_x, right_x], [high_y, high_y], color=c)
        ax.plot([left_x, right_x], [low_y, low_y], color=c)
        ax.plot([left_x, left_x], [low_y, high_y], color=c)
        ax.plot([right_x, right_x], [low_y, high_y], color=c)
        ax.plot(x, y, 'X')


def plot_boxes(imgs, z, width, height, n_sequences=2, future=False, save_path=None):
    """ Pass matching sequene of images and z. """
    # number of images in sequence
    cols = imgs.shape[1]

    # flatten sequences
    imgs = imgs.reshape(-1, *imgs.shape[2:])

    # delete empty color channel from greyscale images
    if imgs.shape[1] == 1:
        imgs = imgs[:, 0]

    fig, axs = plt.subplots(nrows=n_sequences, ncols=cols, figsize=(8, n_sequences*2.5))

    for i, ax in enumerate(axs.flatten()):
        setup_axis(ax)

        ax.imshow(imgs[i].T)

        rect = rectangles_from_z(z[i], ax, width, height)

    plt.suptitle('Future' if future else 'Present')
    plt.subplots_adjust(top=1.0, bottom=0.0,
                        left=0., right=1,
                        wspace=.05, hspace=-.4
                        )
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
    plt.close()


def plot_bg(marginalise_bg, bg_loglik, n_sequences):

    fig, axs = plt.subplots(nrows=n_sequences, ncols=4, figsize=(8, n_sequences*2.5))

    marginalise_bg = marginalise_bg[:, 0]

    for i, ax in enumerate(axs.flatten()):
        setup_axis(ax)

        ax.imshow(marginalise_bg[i].T, vmin=0, vmax=1)

        # overlap is n4, o. just make n4o s.t. indices correspond to patches
        ax.set_title('o: {:.2f}, ll: {:.0f}'.format(marginalise_bg[i].mean(),
                                                    bg_loglik[i]))

    plt.subplots_adjust(top=0.95, bottom=0.0,
                        left=0., right=1,
                        wspace=.05, hspace=.2
                        )

    plt.close()


def plot_patches(patches, marg_patch, overlap, patches_ll, c):
    """For each sequence plot the patches of the first image."""

    fig, axs = plt.subplots(nrows=c.n_plot_sequences, ncols=3, figsize=(8, c.n_plot_sequences*2.5))

    if patches.shape[1] == 1:
        patches = patches[:, 0]

    n = c.num_obj
    if n == 3:
        idxs = [0, 1, 2, 18, 19, 20, 36, 37, 38, 54, 55, 56, 72, 73, 74]
    else:
        raise ValueError

    for idx, ax in zip(idxs, axs.flatten()):
        setup_axis(ax)

        ax.imshow(patches[idx].T, vmin=0, vmax=1)
        ax.imshow(marg_patch[idx].T, alpha=0.6, vmin=0, vmax=1)

        # overlap is n4, o. just make n4o s.t. indices correspond to patches
        ax.set_title('over: {:.2f}, ll: {:.2f}'.format(overlap.flatten()[idx],
                                                       patches_ll.flatten()[idx]))

    plt.subplots_adjust(top=0.95, bottom=0.0,
                        left=0., right=1,
                        wspace=.05, hspace=.2
                        )
    plt.close()


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    plt.figure(figsize=(15, 6))
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad):  # and ("bias" not in n):
            try:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
            except Exception:
                print('No gradients for param {}.'.format(n))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")

    # plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.yscale('log')
    plt.show()
    plt.close()
