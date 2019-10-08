import sys
sys.path.append('..')
from config import VinConfig

class SupVinConfig(VinConfig):

    supairvised = True  # for easy identification in logs
    # Directories
    img_folder = "./img/"  # image folder
    checkpoint_path = None
    log_dir = "./log"

    # Model/training config
    num_epochs = 250
    visual = False  # dont use cnn, learn from states
    num_visible = 6  # Number of visible frames, 2 less than for full model, bc we have no vin overhead
    num_rollout = 8  # Number of rollout frames
    frame_step = 1  # Stepsize when observing frames
    batch_size = 256
    cl = 16  # state code length per object
    # careful: affects magnitude of error
    discount_factor = 0.98  # discount factor for loss from rollouts

    # Data config
    num_episodes = 1000  # The number of episodes
    num_frames = 100  # The number of frames perFalse episode
    width = 32
    height = 32
    channels = 1
    num_obj = 3  # the number of object

    scale_var = 0.3
    pos_var = 0.3
    # bounds for mean of width of bounding box relative to native width
    min_obj_scale = 0.22
    max_obj_scale = 0.3
    # bounds for mean of height of bounding box relative to width
    min_y_scale = 0.99  # 0.75
    max_y_scale = 1.01  # 1.25
    obj_pos_bound = 0.8
    max_threads = 8

    # try to get supairvised running
    use_supair = False
    supair_path = None
    debug_disable_v_error = False
    debug_disable_v_diff_error = False

    # get rid of spike
    debug_rollout = True
    debug_rollout_training = True