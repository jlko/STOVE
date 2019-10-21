# Structured Object-Aware Physics Prediction for Video Modelling and Planning

## Example Animations: Video Prediction and Model-based Control
<div>
    <table width="610" border="0px">
      <tr>
        <th width="100">Real</th>
        <th width="100">STOVE</th>
        <th width="100">VRNN</th>
        <th width="100">SQAIR</th>
        <th width="100">Linear</th>
        <th width="100">Supervised</th>
      </tr>
    </table>
    <img height="100" width="610" src="/figures/comparison_grid_billiards.gif">
    <img height="100" width="610" src="/figures/comparison_grid_gravity.gif">
</div>

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
<br>
<br>
<br>
(a, b) The above depicts the reconstruction and prediction errors of the various models.
The models are given 8 frames of video as input, which they reconstruct. Conditioned on this 
input, all models predict the following 92 frames, which are then compared to ground truth data.
We significantly outperform all baselines in predicting either future images or states in both
scenarios. Additionally, we perform strikingly close to the supervised baseline in the billiards scenario.
10 different sequences of lengths 100 are shown.

(c) The above shows the performance of the compared models in the interactive
environments. The agents control the red ball and negative reward is given
whenever the red ball collides with any of the other balls.
STOVE is used as a world model, predicting future states, frames, and rewards.
MCTS on STOVE can then be used for model-based control.
We compare to MCTS on the real environment states, as well as PPO on the
environment states and raw images.
Again, 10 different sequences of length 100 are shown.

