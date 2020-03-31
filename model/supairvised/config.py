"""Contains configuration class with default values."""
from ..video_prediction.config import StoveConfig

class SupStoveConfig(StoveConfig):
    """Completely specifies an experiment.

    Supairvised scenario refers to state supervision via SuPAIR.
    In supervised scenario states are given from the environment.
    """

    supairvised = True  # for easy identification in logs

    # Model/training config
    num_epochs = 400
    # Number of visible frames, 2 less than for full model bc skip=0
    num_visible = 6
    num_rollout = 8  # Number of rollout frames
    frame_step = 1  # Stepsize when observing frames
    batch_size = 256
    cl = 16  # state code length per object
    # careful: affects magnitude of error
    discount_factor = 0.98  # discount factor for loss from rollouts
    learning_rate = 0.002
    min_learning_rate = 0.0002
    debug_anneal_lr = 20000

    # Data config
    num_obj = 3  # the number of object

    # try to get supairvised running
    load_encoder = None
    debug_disable_v_error = False
    debug_disable_v_diff_error = False

    # get rid of spike
    debug_rollout = True  # run core for directly observed states to init latent
    debug_rollout_training = True  # do the above already during training
    debug_add_noise = False