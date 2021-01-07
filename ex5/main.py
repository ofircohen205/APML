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


def train(data_loader: DataLoader, glo_generator: GeneratorForMnistGLO, writer: SummaryWriter, lr, content_dim, class_dim, epochs, stddev, num_samples):
    content_embeddings = nn.Embedding(num_samples, content_dim)
    classes_embeddings = nn.Embedding(NUMBER_OF_DIGITS, class_dim)

    optimizer = Adam([
        {'params': glo_generator.parameters(), 'lr': lr, 'weight_decay': 1e-4},
        {'params': content_embeddings.parameters(), 'lr': lr, 'weight_decay': 1e-4},
        {'params': classes_embeddings.parameters(), 'lr': lr}
    ])

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    counters = [0, 0]
    for epoch in range(epochs):
        losses = []
        progress = tqdm(total=len(data_loader), desc="epoch % 3d" % epoch)

        for i, (Xi, yi, idx) in enumerate(data_loader):
            content_code = content_embeddings(idx)
            class_code = classes_embeddings(yi)
            content_code_noisy = content_code + torch.randn_like(content_code) * stddev
            class_content_code = torch.cat((class_code, content_code_noisy), 1)

            optimizer.zero_grad()
            rec = glo_generator(class_content_code)
            loss = l1_loss(rec, Xi) + l2_loss(rec, Xi)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            progress.set_postfix({
                'loss': np.mean(losses[-100:])
            })
            progress.update()
            writer.add_scalar('training/loss', np.mean(losses[-100:]), counters[0])
            counters[0] += 1

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
    content_dim = args.content_dim
    class_dim = args.class_dim
    batch_size = args.batch_size
    lr = args.lr
    stddev = args.stddev
    num_samples = args.num_samples

    mnist_class = MNIST(num_samples=num_samples)
    mnist_loader = DataLoader(mnist_class, batch_size=batch_size, shuffle=True)
    generator = GeneratorForMnistGLO(content_dim + class_dim)

    writer = SummaryWriter(log_dir=log_dir)
    train(data_loader=mnist_loader, glo_generator=generator, writer=writer, lr=lr,
          content_dim=content_dim, class_dim=class_dim, epochs=epochs, stddev=stddev, num_samples=num_samples)


if __name__ == '__main__':
    main()
