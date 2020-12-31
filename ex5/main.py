import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import GeneratorForMnistGLO
from datasets import MNIST
from utils import LapLoss, project_l2_ball, show_image


NUMBER_OF_DIGITS = 10


def train(mnist_model: MNIST, mnist_loader: DataLoader, glo_generator: GeneratorForMnistGLO, optimizer, writer: SummaryWriter, batch_size=128, epochs=50, stddev=.3):
    zi = Variable(torch.zeros((batch_size, batch_size)), requires_grad=True)
    class_tensor = torch.from_numpy(np.random.randn(NUMBER_OF_DIGITS, batch_size, batch_size)) * stddev
    content = np.random.randn(len(mnist_loader), batch_size, batch_size) * stddev
    # content = project_l2_ball(content)
    content_tensor = torch.from_numpy(content)
    class_content_tensor = torch.cat((content_tensor, class_tensor), 0).type('torch.FloatTensor')

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    for epoch in range(epochs):
        losses = []
        progress = tqdm(total=len(mnist_loader), desc="epoch % 3d" % epoch)

        for idx, data in enumerate(mnist_loader):
            Xi, yi = data
            Xi = Variable(Xi)
            zi.data = class_content_tensor[idx]

            optimizer.zero_grad()
            # num = np.random.randint(0, batch_size)
            # rec = glo_generator(zi)[idx]
            rec = glo_generator(zi)
            loss = l1_loss(rec, Xi)
            loss += l2_loss(rec, Xi)
            loss.backward()
            optimizer.step()

            # class_content_tensor[idx] = torch.from_numpy(project_l2_ball(zi.data.numpy()))
            class_content_tensor[idx] = torch.from_numpy(zi.data.numpy())
            losses.append(loss.item())
            progress.set_postfix({
                'loss': np.mean(losses[-100:])
            })
            progress.update()

            if idx % 50 == 0:
                generated_images = make_grid(rec)
                writer.add_image(f'training/image', generated_images, epoch)

        progress.close()
# End function


def main():
    log_dir = 'logs'
    batch_size = 128
    if os.path.exists('./data'):
        import torchvision
        mnist = torchvision.datasets.MNIST('./data', train=False, download=False, transform=transforms.ToTensor())
    else:
        mnist = MNIST(2000)

    mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    generator = GeneratorForMnistGLO(128)
    # generator.test()

    writer = SummaryWriter(log_dir=log_dir)
    optimizer = Adam(generator.parameters())
    train(mnist, mnist_loader, glo_generator=generator, optimizer=optimizer, writer=writer, epochs=50, batch_size=128)


if __name__ == '__main__':
    main()
