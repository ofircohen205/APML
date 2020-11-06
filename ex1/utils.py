# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from dataset import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


#######################
###### FUNCTIONS ######
#######################
def inspect_dataset(train_dataset, dev_dataset):
    """
    :param train_dataset:
    :param dev_dataset:
    :return counters_train, counters_dev:
    """
    print("Start Inspecting Dataset")
    train_size = train_dataset.__len__()
    dev_size = dev_dataset.__len__()
    print("Number of train images: {}".format(train_size))
    print("Number of dev images: {}".format(dev_size))
    counters_train = {}
    counters_dev = {}
    for value in label_names().values():
        counters_train[value] = 0
        counters_dev[value] = 0

    for idx in range(train_dataset.__len__()):
        item = train_dataset.__getitem__(idx)
        label = label_names()[item[1]]
        counters_train[label] = counters_train[label] + 1

    for idx in range(dev_dataset.__len__()):
        item = dev_dataset.__getitem__(idx)
        label = label_names()[item[1]]
        counters_dev[label] = counters_dev[label] + 1

    print("Train dataset:")
    for cls in counters_train:
        print("Class {}. size: {}. Percentage: {}".format(cls, counters_train[cls], (counters_train[cls] / train_size)))

    print("Dev dataset:")
    for cls in counters_dev:
        print("Class {}. size: {}. Percentage: {}".format(cls, counters_dev[cls], (counters_dev[cls] / train_size)))

    print("End Inspecting Dataset")
    print("==============================================================================")
    return counters_train, counters_dev
# End function


def create_data_loader(dataset, counters, parameters, init_sampler):
    labels = [label for _, label in dataset]
    class_weights = [dataset.__len__() / counters[label] for label in label_names().values()]
    weights = [class_weights[labels[i]] for i in range(dataset.__len__())]
    if init_sampler:
        sampler = WeightedRandomSampler(weights=weights, num_samples=dataset.__len__())
        data_loader = DataLoader(dataset=dataset, batch_size=parameters['batch_size'], sampler=sampler)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=parameters['batch_size'])
    return data_loader
# End function


def augment_dataset(dataset):
    return None
# End function


def plot(values, title, y_label, x_label):
    plt.plot(values)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(['val'], loc='upper left')
    plt.savefig('./graphs/{}_{}.png'.format(title.replace(' ', '_'), current_time()))
    plt.clf()
# End function


def plot_confusion_matrix_sns(values, title, classes):
    ax = sns.heatmap(values, xticklabels=classes, yticklabels=classes, annot=True,
                     cmap='coolwarm', linewidths=0.5, fmt='d')
    ax.set_ylim(3.0, 0)
    plt.title(title)
    plt.savefig('./graphs/{}_{}.png'.format(title.replace(' ', '_'), current_time()))
    plt.clf()
# End function


def current_time():
    return datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
