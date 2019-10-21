"""Contains configuration class with default values."""

from torch import double


class StoveConfig:
    """Completely specifies an experiment.

    Contains *all* experiment parameters: Data, Training parameters, and
    model parameteres.

    """
    """Experiment Parameters"""
    description = 'unnamed experiment'
    nolog = False
    experiment_dir = './experiments/stove/unsorted'
    checkpoint_path = None  # if set to valid checkpoint, said checkpoint is loaded
    load = False  # Load parameters from checkpoint file
    action_conditioned = None  # set by main.py depending on data loaded
    random_seed = None  # seed for random spn region graph (auto set in main.py)
    supairvised = False
    load_encoder = None  # if set to a valid checkpoint, supair from checkpoint is loaded
    supair_only = False  # if enabled, only supair is trained
    supair_grad = True  # if disabled, parameters of supair are not optimised
    debug_test_mode = False  # awesome flag, use for debugging.
    # (essentially sets batch_size to 1, makes calling train() for debugging fast)

    """Data Parameters"""
    traindata = './data/billiards_train_data.pkl'
    testdata = './data/billiards_test_data.pkl'
    num_visible = 8  # 6  # Number of visible frames per sequence.
    num_rollout = 8  # Number of rollout frames (currently, only used in testing).
    frame_step = 1  # Stepsize when observing frames
    num_episodes = 1000  # no. of generated sequences
    num_frames = 100  # no. of frames per sequence
    width = None  # auto set in main.py
    height = None  # auto set in main.py
    channels = 1  # debug_bw is enabled
    num_obj = 3  # the number of object
    r = None  # object radius, for rollout rendering (auto set in main.py)
    coord_lim = None  # max of true env states, for error (auto set in main.py)
    action_space = None  # dim of action_space, for error (auto set in main.py)

    """Training Configuration"""
    batch_size = 256
    cl = 32
    learning_rate = 0.002  # see adjust_learning_rate() in train.py
    min_learning_rate = 0.0002  # see adjust_learning_rate() in train.py
    debug_anneal_lr = 40000.0  # see adjust_learning_rate() in train.py
    num_epochs = 400
    debug_amsgrad = True
    debug_gradient_clip = True  # clip gradients

    # Pytorch Configuration
    """(Some of these are dynamically set in main.py upon initialisation.)"""
    device = None
    dtype = double
    max_threads = 8
    num_workers = 4

    # Logging Configuration
    debug = True  # enables logging of position errors etc.
    n_plot_sequences = 5  # no. of sequences to plot
    print_every = 100
    plot_every = 1e19  # basically disable plotting
    save_every = 10000
    visdom = False
    rollout_idxs = [0, 10, 20, 30, 40, 100]  # idx of testset for rollout
    debug_extend_plots = False

    """Model Config: Stove"""
    # stds on generative dynamics
    skip = 2
    transition_lik_std = [0.01, 0.01, 0.01, 0.01]
    debug_fix_supair = True  # smooth over supair states
    # use appearance for matching
    debug_match_appearance = False
    debug_no_latents = False  # will use for later experiments in thesis

    """Model Config: Action-Conditioned"""
    # Following parameters concern action-conditioned video prediction.
    debug_reward_factor = 15000  # reward tradeoff for combined loss
    debug_reward_rampup = 20000  # no. of steps during which reward loss lin. increases
    debug_mse = False
    # add object appearance to core
    debug_core_appearance = False
    debug_appearance_dim = 3

    """Model Config: Dynamics"""
    debug_nonlinear = 'relu'  # or 'elu', nonlinearity used in dynamics core
    debug_latent_q_std = 0.04  # bound on inferred std on latent
    # Initalisation
    debug_xavier = False  # use xavier initialisation in model

    """Model Config: SPN"""
    debug_bw = True  # do not model colors in SPN
    # no. of pixels used to model objects in SPN
    patch_height = 10
    patch_width = 10
    # bounds on variance for leafs in object SPN
    obj_min_var = 0.12
    obj_max_var = 0.35
    # bounds on variance for leafs in background SPN
    bg_min_var = 0.002
    bg_max_var = 0.16
    # scale and position variance for parameter distributions
    scale_var = 0.3  # DEBUG 0.003
    pos_var = 0.3
    # bounds for mean of width of bounding box relative to canvas width
    min_obj_scale = 0.1
    max_obj_scale = 0.8
    # bounds for mean of height of bounding box relative to width
    min_y_scale = 0.75  # 0.75
    max_y_scale = 1.25  # 1.25
    # max and min object position
    obj_pos_bound = 0.9
    # number of leaf distributions and sums
    obj_spn_num_gauss = 10
    obj_spn_num_sums = 10
    # parameter for penalty term in p(z), discouraging overlap
    overlap_beta = 10.0
    # (Trying out new things. Debugging code.)
    # Replace SPNs
    debug_bg_model = False  # use simple gauss for background instead of SPN
    debug_obj_spn = False  # use simple gauss for objects instead of SPN
    debug_simple_bg_var = 0.1  # debug 0.35
    debug_simple_obj_var = 0.2
