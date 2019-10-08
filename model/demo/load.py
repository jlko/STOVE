# move data and experiments into project folder on 04_rl
from vin import main

restore = '/path/to/run/folder/'
extras = {
        'nolog': True, 'traindata': '/path/to/billards_w_actions_train_data.pkl',
        'testdata': '/path/to/billards_w_actions_test_data.pkl'
        }
trainer = main(extras=extras, restore=restore)

model = trainer.net


# this is how to call

# once per timestep
elbo, prop_dict, rewards = self.net(
        present, # batch, frame_stack (currently 8), c=3, w, h
        0, # TODO: this fills prop dict and needs to be set according to config plotting thingy
        action, # aligned present
        False
        )

# states are stored in prop_dict['z']
rewards#, all past #.view() == batch, frame-2, 1 are interesting



z_pred, rewards_pred = self.net.rollout(
        prop_dict['z'][:, -1],
        num=rollout_length,  # number of future steps
        actions=future_actions,  # batch, num, 9
        appearance=prop_dict['obj_appearances'][:, -1] # 
        )
