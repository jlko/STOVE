import torch


class VinConfig:
    description = 'unnamed experiment'
    nolog = False

    # Directories
    traindata = './data/billards_new_env_train_data.mat'
    testdata = './data/billards_new_env_test_data.mat'
    experiment_dir = './experiments/unsorted'
    checkpoint_path = None  # if set to valid checkpoint, said checkpoint is loaded

    # Model/training config
    load = False  # Load parameters from checkpoint file
    num_visible = 8  # 6  # Number of visible frames
    num_rollout = 8  # Number of rollout frames
    frame_step = 1  # Stepsize when observing frames
    batch_size = 256  # DEBUG. Allow for some learning by doing more updates?
    cl = 32  # 16 DEBUG. fix to 8 while we only do z  # state code length per object
    discount_factor = 0.98  # discount factor for loss from rollouts
    learning_rate = 0.002  # debug 0.0005 or 0.001
    overlap_beta = 10.0  # Parameter for penalty term in p(z), discouraging overlap
    device = None  # set in vin.py
    num_epochs = 400
    # set by main.py depending on data loaded
    # enables or disables action_conditioned prediction
    action_conditioned = None
    supairvised = False    

    save_every = 2000
    v_mode = 'from_state'  # how to get speeds: ['from_state' or 'from_imgs']
    skip = 2 if v_mode == 'from_state' else 3 if v_mode == 'from_img' else 'error'

    load_supair = None  # None  # if set to a valid checkpoint, supair from checkpoint is loaded
    supair_only = False  # if enabled, only supair is trained
    supair_grad = True  # if disabled, parameters of supair are not optimised

    # Data config
    num_episodes = 1000  # The number of episodes
    num_frames = 100  # DEBUG 100  # The number of frames per episode
    width = 32  # auto set in VIN
    height = 32  # auto set in VIN
    channels = 1  # debug_bw is enabled
    num_obj = 3  # the number of object

    # SPN Configs
    patch_height = 10
    patch_width = 10
    # Bounds on variance for leafs in object SPN
    obj_min_var = 0.12
    obj_max_var = 0.35  # DEBUG 0.35
    # Bounds on variance for leafs in background SPN
    bg_min_var = 0.002  # DEBUG 0.002
    bg_max_var = 0.16  # 0.12 DEBUG 0.16   # DEBUG from crazyKs new sprites git bg_max_var = 0.1
    # Sampling Parameters
    # Scale and position variance for parameter distributions
    scale_var = 0.3  # DEBUG 0.003
    pos_var = 0.3
    # bounds for mean of width of bounding box relative to native width
    min_obj_scale = 0.1
    max_obj_scale = 0.3
    # bounds for mean of height of bounding box relative to width
    min_y_scale = 0.99  # 0.75
    max_y_scale = 1.01  # 1.25
    # max and min object position (center)
    obj_pos_bound = 0.8
    obj_spn_num_gauss = 10
    obj_spn_num_sums = 10

    # VIN Params
    transition_lik_std = [0.01, 0.01, 0.01, 0.01]

    # Plot config
    n_plot_sequences = 5  # no. of sequences to plot
    print_every = 100
    plot_every = 1e19  # basically disable plotting, I never look at them and they take up space
    visdom = False

    # Pytorch
    dtype = torch.double
    max_threads = 8

    # Plot options
    # Indices for which to create rollouts.
    rollout_idxs = [0, 10, 20, 30, 40, 100]

    # Debug
    num_workers = 4
    debug = True
    debug_amsgrad = True
    debug_extend_plots = False
    encoder = 'rnn'  # rnn or 'cnn'
    random_seed = None

    debug_nonlinear = 'relu'  # or 'elu'

    # Debug Options 
    debug_color = False
    debug_bw = True
    debug_noise = False

    debug_bg_model = False
    debug_obj_spn = False
    debug_spn_mean = False
    debug_spn_bg_max_mean = None
    debug_spn_obj_min_mean = None

    debug_simple_bg_var = 0.1  # debug 0.35
    debug_simple_obj_var = 0.2

    debug_xavier = False
    debug_greedy_matching = True
    debug_fix_supair = True 
    debug_gradient_clip = True

    data = 'billiards'  # 'multi_mnist'  # or billiards

    debug_cnn = True

    debug_analytic_kl = False
    # debug_supervised or debug_rollout_loss need to be true
    debug_sample_rollout = False  # sample during training in rollout
    debug_code = False
    debug_fixed_std = None
    debug_test_mode = False
    debug_no_latents = False  # will use for later experiments in thesis

    debug_anneal_lr = 40000.0
    debug_rollout_loss_num = 5
    debug_latent_p = True
    debug_latent_q_std = 0.04
    debug_offset = 'old'

    # set by load_dataa
    r = None
    coord_lim = None
    action_space = None

    debug_match_v_too = False

    debug_reward_factor = 15000
    debug_reward_rampup = 20000

    debug_reward_head = 'old'
    debug_mse = False

    # add object appearance to core
    debug_core_appearance = False
    debug_appearance_dim = 3
    # use appearance for matching
    debug_match_appearance = False