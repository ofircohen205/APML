# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from dataset import *
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


#######################
###### FUNCTIONS ######
#######################
def inspect_dataset(train_df, dev_df):
    """
    :param train_df:
    :param dev_df:
    :return:
    """
    train_size = train_df.__len__()
    dev_size = dev_df.__len__()
    print("Number of train images: {}".format(train_size))
    print("Number of dev images: {}".format(dev_size))
    counters_train = {}
    counters_dev = {}
    for value in label_names().values():
        counters_train[value] = 0
        counters_dev[value] = 0

    for idx in range(train_df.__len__()):
        item = train_df.__getitem__(idx)
        label = label_names()[item[1]]
        counters_train[label] = counters_train[label] + 1

    for idx in range(dev_df.__len__()):
        item = dev_df.__getitem__(idx)
        label = label_names()[item[1]]
        counters_dev[label] = counters_dev[label] + 1

    print("Train dataset:")
    for cls in counters_train:
        print("Class {}. size: {}. Percentage: {}".format(cls, counters_train[cls], (counters_train[cls] / train_size)))

    print("Dev dataset:")
    for cls in counters_dev:
        print("Class {}. size: {}. Percentage: {}".format(cls, counters_dev[cls], (counters_dev[cls] / train_size)))
# End function


def evaluate_model(train_df, dev_df, epochs=2):
    """
    :param train_df:
    :param dev_df:
    :param epochs:
    :return:
    """
    model = SimpleModel()
    model.load(path='./data/pre_trained.ckpt')
    model.eval()
    val_losses = []
    loss_fn = nn.CrossEntropyLoss()
    dev_loader = DataLoader(dataset=dev_df, batch_size=16, shuffle=True)
    for epoch in range(epochs):
        with torch.no_grad():
            for x_val, y_val in dev_loader:
                label_pred = model(x_val)
                val_loss = loss_fn(label_pred, y_val)
                val_losses.append(val_loss.item())

    plot(val_losses)
    print(model.state_dict())
# End function


def plot(values):
    plt.plot(values)
    plt.title("model evaluation")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(['val'], loc='upper left')
    plt.show()
# End function


def main():
    train_df = get_dataset_as_torch_dataset(path='./data/train.pickle')
    dev_df = get_dataset_as_torch_dataset(path='./data/dev.pickle')
    # Inspect dataset:
    inspect_dataset(train_df, dev_df)
    # Evaluate given model
    evaluate_model(train_df, dev_df, 2)
# End function


if __name__ == '__main__':
    main()
