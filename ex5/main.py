from tqdm import tqdm
import numpy as np
import torchvision

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import GeneratorForMnistGLO
from datasets import MNIST
from utils import parse_args, query_yes_no
import os, shutil


NUMBER_OF_DIGITS = 10


def train(data_loader: DataLoader, glo_generator: GeneratorForMnistGLO, optimizer, writer: SummaryWriter, batch_size=128, code_dim=128, epochs=50, stddev=.3):
    zi = Variable(torch.zeros((batch_size, code_dim)), requires_grad=True)
    class_tensor = torch.normal(mean=0, std=stddev, size=(NUMBER_OF_DIGITS, batch_size, code_dim))
    content_tensor = torch.normal(mean=0, std=stddev, size=(len(data_loader), batch_size, code_dim))
    class_content_tensor = torch.cat((content_tensor, class_tensor), 0).type('torch.FloatTensor')

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    counters = [0, 0]
    for epoch in range(epochs):
        losses = []
        progress = tqdm(total=len(data_loader), desc="epoch % 3d" % epoch)

        for idx, (Xi, yi) in enumerate(data_loader):
            zi.data = class_content_tensor[idx]

            optimizer.zero_grad()
            rec = glo_generator(zi)
            loss = l1_loss(rec, Xi) + l2_loss(rec, Xi)
            loss.backward()
            optimizer.step()

            class_content_tensor[idx] = zi.data
            losses.append(loss.item())
            progress.set_postfix({
                'loss': np.mean(losses[-100:])
            })
            progress.update()
            writer.add_scalar('training/loss', np.mean(losses[-100:]), counters[0])
            counters[0] += 1

            if idx % 50 == 0:
                generated_images = make_grid(rec)
                ground_truth_images = make_grid(Xi)
                writer.add_image('training/image/generated', generated_images, counters[1])
                writer.add_image('training/image/ground_truth', ground_truth_images, counters[1])
                counters[1] += 1

        progress.close()
# End function


def main():
    args = parse_args()
    args.log_dir = os.path.join(args.log_dir, args.name)

    if os.path.exists(args.log_dir):
        if query_yes_no('You already have a run called {}, override?'.format(args.name)):
            shutil.rmtree(args.log_dir)
        else:
            exit(0)

    del args.__dict__['name']
    log_dir = args.log_dir
    epochs = args.epochs
    code_dim = args.code_dim
    batch_size = args.batch_size
    lr = args.lr
    stddev = args.stddev

    mnist_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    mnist_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    generator = GeneratorForMnistGLO(code_dim)

    writer = SummaryWriter(log_dir=log_dir)
    optimizer = Adam(generator.parameters(), lr=lr)
    train(data_loader=mnist_loader, glo_generator=generator, optimizer=optimizer, writer=writer, batch_size=batch_size,
          code_dim=code_dim, epochs=epochs, stddev=stddev)


if __name__ == '__main__':
    main()
