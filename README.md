# Structured Object-Aware Physics Prediction for Video Modelling and Planning
See [arxiv.org/1910.02425](https://arxiv.org/abs/1910.02425) or [openreview.net/forum?id=B1e-kxSKDH](https://openreview.net/forum?id=B1e-kxSKDH) for further information.

## Example Animations
### Video Prediction
<div>
    <table width="710" border="0px">
      <tr>
        <th width="100">Real</th>
        <th width="100">Ours</th>
        <th width="100">VRNN</th>
        <th width="100">SQAIR</th>
        <th width="100">DDPAE</th>
        <th width="100">Linear</th>
        <th width="100">Supervised</th>
      </tr>
    </table>
    <img height="100" width="710" src="/figures/comparison_grid_billiards.gif">
</div>


<div>
    <table width="710" border="0px">
      <tr>
        <th width="100">Real</th>
        <th width="100">Ours</th>
        <th width="100">VRNN</th>
        <th width="100">SQAIR</th>
        <th width="100">DDPAE</th>
        <th width="100">Linear</th>
        <th width="100">Supervised</th>
      </tr>
    </table>
    <img height="100" width="710" src="/figures/comparison_grid_gravity.gif">
</div>


The above depicts the reconstruction and prediction errors of the various models.
The models are given 8 frames of video as input, which they reconstruct. Conditioned on this 
input, all models predict the following 92 frames.
Only STOVE manages to generate visually convincing physically behavior over longer timeframes.
10 different sequences of length 100 are shown.

### Model-Based Control
<div>
    <table width="410" border="0px" style="font-size:8">
      <tr>
        <th width="100">MCTS on STOVE</th>
        <th width="100">MCTS on Real Env.</th>
        <th width="100">PPO on Env. States</th>
        <th width="100">PPO on Env. Images</th>
      </tr>
    </table>
    <img height="100" width="410" src="/figures/comparison_grid_planning.gif">
</div>


The above shows the performance of the compared models in the interactive
environments. The agent controls the red ball and negative reward is given
whenever the red ball collides with any other ball.
STOVE is used as a world model, predicting future states, frames, and rewards.
MCTS can then be used on STOVE for model-based control.
We compare to MCTS on the real environment states, as well as PPO on the
environment states and raw images.
Again, 10 different sequences of length 100 are shown.

## Abstract
>Humans can easily predict future outcomes even in settings
with complicated interactions between objects. For computers, however,
learning models of interactions from videos in an unsupervised fashion is hard. 
In this paper, we demonstrate that structure and compositionality
are key to solving this problem. Specifically, we develop a novel
video prediction model from two distinct components: a scene model
and a physics model. Both models are compositional and exploit repeating
elements wherever possible. We impose a highly structured bottleneck
between model components to allow for fast learning and clearly
interpretable functionality, without losing any generality or performance.
This fully compositional approach yields a strong video prediction
model, which clearly outperforms relevant baselines.
We produce realistic looking physical behaviour over a possibly
infinite time frame and perform competitively even compared to a 
supervised approach.
Finally, we demonstrate the strength of our model as a simulator for
sample efficient model-based reinforcement learning in tasks with
heavily interacting objects.

## Other animations
<img height="100" width="100" src="/figures/long_rollout.gif">
STOVE's rollouts are stable for a possibly infinite number of timesteps.
(Shown are 2000 frames of rollout and we tested up to 100000.)
<img height="100" width="100" src="/figures/multiball_billiards.gif">
All components of STOVE scale well to videos with larger number of objects!

## Citation
Please cite our work here and at [arxiv.org/1910.02425](https://arxiv.org/abs/1910.02425) as 
```
@inproceedings{kossen2020structured,
  title={Structured Object-Aware Physics Prediction for Video Modeling and Planning},
  author={Kossen, Jannik and Stelzner, Karl and Hussing, Marcel and Voelcker, Claas and Kersting, Kristian},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2020}
}
```

## Data
Run `run_scripts.py --create-data` to generate billiards and gravity data.
Also random data collected in the RL setting to train the action-
conditioned world model is generated.

## Model
Run `bash run_files/run_models.sh` to train the model on billiards and gravity data.
Also an actioned-conditioned world model is trained on the avoidance task.


## Interactive
`python run_scripts.py --interactive` allows you to either play in a live
environment or a model simulation of the environment.

## Questions
If you have any questions or problems regarding the code or paper do not hesitate to contact us.
