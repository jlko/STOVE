# This file executes demo runs of all STOVE modalities

# Stove on Billiards Data
python run_stove.py --args \
    description billiards_example_run \
    experiment_dir ./experiments/stove/00_billiards_example \
    traindata ./data/billiards_train.pkl \
    testdata ./data/billiards_test.pkl

# Stove on Gravity Data
python run_stove.py --args \
    description gravity_example_run \
    experiment_dir ./experiments/stove/01_gravity_example \
    traindata ./data/gravity_train.pkl \
    testdata ./data/gravity_test.pkl

# Action Conditioneed Stove on Avoidance Task on Billiards Data
python run_stove.py --args \
    description action_conditioned_example \
    experiment_dir ./experiments/stove/02_action_conditioned \
    traindata ./data/avoidance_train.pkl \
    testdata ./data/avoidance_test.pkl \
    debug_core_appearance True \
    debug_match_appearance False \
    num_epochs 500
