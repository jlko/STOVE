import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import visdom

from rat_torch import RatSpn, SpnArgs
import region_graph

vis = visdom.Visdom()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x.view((28 * 28,)))])

batch_size = 256

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)


rg = region_graph.RegionGraph(range(28 * 28))
for _ in range(0, 8):
    rg.random_split(2, 2)

args = SpnArgs()
args.num_sums = 20
args.num_gauss = 10
spn = RatSpn(1, region_graph=rg, name="spn", args=args)
spn.num_params()

criterion = nn.CrossEntropyLoss()
# print(list(spn.parameters()))
optimizer = optim.Adam(spn.parameters())

for epoch in range(20):
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs
        labels = labels
        optimizer.zero_grad()
        outputs, child_acts  = spn.compute_activations(inputs,
                                                       get_sum_child_acts=True)
        outputs = outputs[spn.output_vector]
        # loss = criterion(outputs, labels)
        loss = -torch.mean(outputs)
        loss.backward()
        optimizer.step()

        prediction = torch.argmax(outputs, 1)
        correct += sum(prediction == labels).item()

        running_loss += loss.item()
        if i % 600 == 0:
            print('[%d, %5d] loss: %.3f acc %.2f' %
                  (epoch + 1, i + 1, running_loss / 600, correct / (600 * batch_size)))
            running_loss = 0.0
            correct = 0
            recons = []
            for j in range(8):
                max_idxs = {vec: np.argmax(p[j].detach().numpy(), 0)
                            for (vec, p) in child_acts.items()}
                recon = spn.reconstruct(max_idxs, 0, False)
                recons.append(recon)
            recons = np.clip(np.stack(recons, 0), 0., 1.)
            vis.images(np.reshape(inputs[:8], (8, 1, 28, 28)), padding=4)
            vis.images(np.reshape(recons, (8, 1, 28, 28)), padding=4)
            torch.save(spn, './demo/spn-ceckpoint')
print('done')


