# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os


#####################
###### CLASSES ######
#####################
class Trainer:
    """
    Trainer class for training a given model
    """
    def __init__(self, model: SimpleModel, dataset: DataLoader, criterion, lr: float, betas: list,
                 epochs: int, batch_size: int, num_classes: int, epsilon: float, name: str, path: str):
        """
        :param model: the model we wish to evaluate
        :param dataset: the data we wish to evaluate our model on
        :param criterion: loss function
        :param lr: learning rate for optimizer
        :param betas: betas for optimizer
        :param epochs: number of epochs
        :param batch_size: what is the size of the batched inputs
        :param num_classes: number of classes the model classifies
        :param epsilon: if running_loss < epsilon then save the model_state
        :param name: for saving the model_state and plots
        """
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
        self.ckpt = os.path.join(path, "{}.ckpt".format(self.name))
        self.model_states = []
        self.losses = []
        self.accuracies = []

    def __train__(self, save):
        total = 0.0
        correct = 0.0
        optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.betas[0], weight_decay=1e-3)
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for idx, data in enumerate(self.dataset, 0):
                running_loss, total, correct = self.__fit_predict__(data, optimizer, running_loss, total, correct)

                if idx % 200 == 199:
                    self.__collect_data__(epoch, idx, running_loss, save, total, correct)
                    running_loss = 0.0
                    total = 0.0
                    correct = 0.0

    def __fit_predict__(self, data, optimizer, running_loss, total, correct):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        return running_loss, total, correct

    def __collect_data__(self, epoch, idx, running_loss, save, total, correct):
        running_loss = running_loss / 200
        if save and running_loss < self.epsilon:
            self.model_states.append({
                'state_dict': self.model.state_dict(),
                'loss': running_loss
            })
        self.losses.append(running_loss)
        self.accuracies.append(correct / total)
        print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss))
# End class
