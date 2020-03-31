# This file contains the executables for an unordered selection of experiments.

# Sup(er/air)vised experiments
# ============================

# Run supervised model for ablation.
# python run_scripts.py --supervised \
#     description supervised \
#     experiment_dir ./experiments/stove/03_supairvised \
#     traindata /home/jkossen/share/billards_new_env_train_data.mat \
#     testdata /home/jkossen/share/billards_new_env_test_data.mat \
#     num_epochs 250

# Run supairvised model for ablation
# python run_scripts.py --supervised \
#     description vanilla_supair_no_rollout \
#     experiment_dir ./experiments/supairvised/00_from_supair \
#     load_encoder ./experiments/supair/00_vanilla/run001/checkpoint \
#     supair_grad False \
#     num_rollout 1

# python run_scripts.py --supervised \
#     description vanilla \
#     experiment_dir ./experiments/supairvised/00_from_supair \
#     load_encoder ./experiments/supair/00_vanilla/run001/checkpoint \
#     supair_grad False 

# python run_scripts.py --supervised \
#     description vanilla_with_grad \
#     experiment_dir ./experiments/supairvised/00_from_supair \
#     load_encoder ./experiments/supair/00_vanilla/run001/checkpoint \
#     supair_grad True

# Ablations
# =========

# Run model without latent space
# python run_stove.py --args \
#     description disable_latents \
#     experiment_dir ./experiments/stove/10_no_latents_gravity \
#     debug_no_latents True \
#     traindata /home/jkossen/share/new_gravity_train_data.pkl \
#     testdata /home/jkossen/share/new_gravity_test_data.pkl 

# Run model without velocity
# python run_stove.py --args \
#     description disable_velocity \
#     experiment_dir ./experiments/stove/09_no_latents \
#     debug_no_velocity True \
#     traindata ./data/gravity_train.pkl \
#     testdata ./data/gravity_test.pkl

# Run model without reusing latent space
# python run_stove.py --args \
#     description disable_reuse \
#     experiment_dir ./experiments/stove/09_no_latents \
#     debug_no_reuse True

# Test hypothesis that gradient of reusing dynamics model for inference is
# essential.
# python run_stove.py --args \
#     description gradient_hypothesis \
#     experiment_dir ./experiments/stove/05_reuse_dynamics \
#     debug_no_reuse True 

# Multiball
# =========

# Test Performance of model on 6 instead of 3 balls.
# Tighter obj_scale was necessary to avoid local minima

# python run_stove.py --args \
#     description multi_ball \
#     experiment_dir ./experiments/stove/04_multi_ball \
#     traindata ./data/multibilliards_train.pkl \
#     testdata ./data/multibilliards_test.pkl \
#     debug_match_objects greedy \
#     print_every 1000 \
#     overlap_beta 100 \
#     max_obj_scale 0.22


# Energies
# ========

# Run model on data with differing energies
# python run_stove.py --args \
#     description multi_energy_v_2.0 \
#     experiment_dir ./experiments/stove/08_multi_energy \
#     traindata ./data/billiards_energy_2.0_train.pkl \
#     testdata ./data/billiards_energy_2.0_test.pkl \
#     checkpoint_path ./experiments/stove/08_multi_energy/run001/checkpoint


# Investigations into Energy Constancy
# ====================================

# Investigate STOVE energies over training
# TEST="debug_test_mode False"

# python run_scripts.py --supervised \
#     description more_epochs_better_energy \
#     experiment_dir ./experiments/supervised/baseline_billiards \
#     num_epochs 800 \
#     $TEST


# Add noise to supervise for effect on energy prediction.
# python run_scripts.py --supervised \
#     description no_noise_baseline \
#     experiment_dir ./experiments/supervised/energy_noise \
#     debug_add_noise True \
#     $TEST

# Dynamics Core
# =============

# Try out other architectures for dynamics core.
# Especially w.r.t. action-conditioned setting, where we suspect that the
# architecture is not ideally suited to modelling independently moving objects.
# TEST="debug_test_mode False"

# Supervised Action-Conditioned baseline.
# cd ../any/
# git checkout 10_new_self_dyn_1
# python run_scripts.py --supervised \
#     description simpler_attention_2paths \
#     experiment_dir ../development/experiments/supervised/action_conditioned \
#     traindata ./data/avoidance_train.pkl \
#     testdata ./data/avoidance_test.pkl \
#     num_epochs 500 \
#     $TEST
# cd .../development/

# Modification to self propagation part.
# cd ../any/
# git checkout 08_simple_core_mode
# python run_scripts.py --supervised \
#     description simple_core_mod \
#     experiment_dir ../development/experiments/supervised/baseline_billiards \
#     $TEST
# cd ../development

# Test model with multiple environments at the same time and check performance.
# cd ../any/
# git checkout 11_multi_env
# python run_scripts.py --supervised \
#     traindata ./data/billiards_gravity_train.pkl \
#     testdata ./data/billiards_gravity_test.pkl \
#     debug_core_appearance True \
#     debug_match_appearance False \
#     description naive_no_indicator \
#     debug_disable_multi_env_indicator True \
#     experiment_dir ../development/experiments/supervised/multi_env
# cd ../development/
