# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from dataset import *
from utils import *
from sklearn.metrics import *
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn


#####################
###### CLASSES ######
#####################
class Trainer:
    """
    Trainer class for training a given model
    """
    def __init__(self, model: SimpleModel, dataset: DataLoader, criterion, lr, betas, epochs, batch_size, num_classes, epsilon, name):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.betas = betas
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.num_classes = num_classes
        self.name = name
        self.epsilon = epsilon
        self.model_states = []
        self.losses = []
        self.accuracies = []
        self.mislabeled = []

    def __train__(self, dataset, save):
        print("Start Train Model")
        total = 0.0
        correct = 0.0
        index = 0
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, betas=self.betas)
        self.model.train()
        for epoch in range(self.epochs):
            last_running_loss = 0.0
            running_loss = 0.0
            for idx, data in enumerate(self.dataset, 0):
                index += 1
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.mislabeled.append({'idx': index, 'loss': loss.item()})
                _, predicted = torch.max(outputs.data, 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total += labels.size(0)
                correct = (predicted == labels).sum().item()

                if idx % 200 == 199:
                    running_loss = running_loss / 200
                    last_running_loss = running_loss
                    if running_loss < self.epsilon and running_loss - last_running_loss < 0.1:
                        self.model_states.append({
                            'state_dict': self.model.state_dict(),
                            'loss': running_loss
                        })
                    self.losses.append(running_loss)
                    self.accuracies.append(correct / total)
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss))
                    running_loss = 0.0
                    total = 0.0
                    correct = 0.0

        if save:
            self.model_states = sorted(self.model_states, key=lambda x: x['loss'])
            torch.save({'model_state_dict': self.model_states[0]['state_dict']}, './models/trained_{}.ckpt'.format(current_time()))
            plot(self.losses, "{} Loss".format(self.name), "loss", "epoch")
            plot(self.accuracies, "{} Accuracy".format(self.name), "accuracy", "epoch")

        sorted(self.mislabeled, key=lambda x: x['loss']).reverse()
        print("==============================================================================")
# End class
