from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from models import GeneratorForMnistGLO
from datasets import MNIST
from laploss import LapLoss


def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)
# End function


def train(mnist_model: MNIST, glo_generator: GeneratorForMnistGLO, optimizer, epochs, batch_size, code_dim):
    zi = Variable(torch.zeros((batch_size, code_dim)), requires_grad=True)
    Z = np.random.randn(len(mnist_model.data), code_dim, code_dim) * 0.3
    Z = project_l2_ball(Z)
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    for epoch in range(epochs):
        losses = []
        progress = tqdm(total=len(mnist_model.data), desc="epoch % 3d" % epoch)

        for i, (Xi, yi, idx) in enumerate(mnist_model.data):
            Xi = Variable(torch.from_numpy(Xi))
            zi.data = torch.FloatTensor(Z[idx])

            optimizer.zero_grad()
            num = np.random.randint(0, code_dim)
            rec = glo_generator(zi)[num]
            loss = l1_loss(rec, Xi)
            loss += l2_loss(rec, Xi)
            loss.backward()
            optimizer.step()

            Z[idx] = project_l2_ball(zi.data.numpy())
            losses.append(loss.item())
            progress.set_postfix({
                'loss': np.mean(losses[-100:])
            })
            progress.update()

            if i % 100 == 0:
                mnist_model.present(rec[0].detach().numpy())

        progress.close()
# End function


def main():
    mnist = MNIST(2000)
    generator = GeneratorForMnistGLO(128)

    # criterion = LapLoss()
    optimizer = Adam(generator.parameters())
    train(mnist, generator, optimizer, 50, 128, 128)


if __name__ == '__main__':
    main()
