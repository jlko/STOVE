from vin import main
import train
import numpy as np
import visdom

vis = visdom.Visdom(port=8999)

restore = './nice_result/run005/'

trainer = main(restore=restore)

obj_spn = trainer.net.obj_spn

params = obj_spn.get_sum_params()

max_idxs = {vec: np.argmax(p.detach().numpy(), 0)
            for (vec, p) in params.items()}

mpe_img = obj_spn.reconstruct(max_idxs, 0, sample=False)
mpe_img = np.clip(mpe_img, 0., 1.)
print(mpe_img.min(), mpe_img.max(), mpe_img.mean())
vis.image(np.reshape(mpe_img, (10, 10)))
