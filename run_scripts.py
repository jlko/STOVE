"""Run any of the other package capabilities using execution flags.

First argument has to specify, which script is to be executed.
Any other arguments will be passed to said script.
"""
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Live play in environment.
    parser.add_argument('--interactive', dest='interactive',
                        action='store_true')

    # Load trained model and get conditionally generated frames of rollout.
    parser.add_argument('--pixel-error', dest='pixel_error',
                        action='store_true')

    # Create standard collection of data sets.
    parser.add_argument('--create-data', dest='create_data',
                        action='store_true')

    # Run the MCTS either on a pretrained STOVE instance or the environment.
    parser.add_argument('--mcts', dest='mcts', action='store_true')

    # Runs the integrated loop training of MCTS and STOVE.
    parser.add_argument('--mcts-loop', dest='mcts_loop', action='store_true')

    # Runs a completely random baseline of the environment agent.
    parser.add_argument('--random-baseline', dest='random_baseline',
                        action='store_true')

    # Run supervised ablation
    parser.add_argument('--supervised', dest='supervised',
                        action='store_true')


    args, _ = parser.parse_known_args()
    script_args = sys.argv[2:]
    print(sys.argv)
    print(script_args)

    if args.interactive:
        from scripts import interactive

        interactive.main(script_args)

    if args.pixel_error:
        from scripts import pixel_error
        pixel_error.main(script_args)


    if args.create_data:
        from model.envs import envs

        envs.main(script_args)

    if args.mcts:
        from scripts import run_mcts
        run_type = script_args[0]
        run_name = script_args[1]

        if run_type == 'STOVE':
            mcts_args = {k: eval(v) for k, v in zip(script_args[3::2], script_args[4::2])}
            print(mcts_args)
            run_restore_point = script_args[2]
            run_mcts.main_mcts_model(run_name, run_restore_point, **mcts_args)
        elif run_type == 'ENV':
            mcts_args = {k: eval(v) for k, v in zip(script_args[2::2], script_args[3::2])}
            print(mcts_args)
            run_mcts.main_mcts_env(run_name, **mcts_args)

    if args.mcts_loop:
        from scripts import mcts_loop_training
        loop_args = dict(zip(script_args[0::2], script_args[0::2]))
        mcts_loop_training.train_loop(**loop_args)

    if args.random_baseline:
        from model.mcts.random_env_baseline import main_mcts_loop

        run_name = script_args[0]
        main(run_name)

    if args.supervised:
        from model import main
        # load any additional args if provided
        supervised_args = dict(zip(script_args[0::2], script_args[1::2]))
        supervised_args.update({'supairvised': True})
        trainer = main.main(sh_args=supervised_args)
        trainer.train()

    print("End of run_scripts.")
