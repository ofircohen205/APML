# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from dataset import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import *
from statistics import mean
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle


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


def evaluate_model(model, data_loader, counters_dev):
    """
    :param counters_dev:
    :param model:
    :param data_loader:
    """
    print("Start Evaluate Model")
    f1_scores, recalls, precisions = [], [], []
    num_of_classes = label_names().__len__()
    class_correct = list(0. for i in range(num_of_classes))
    class_total = list(0. for i in range(num_of_classes))

    with torch.no_grad():
        for idx, data in enumerate(data_loader, 0):
            model.eval()

            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            precisions.append(precision_score(labels, predicted, average='weighted', zero_division=1))
            recalls.append(recall_score(labels, predicted, average='weighted'))
            f1_scores.append(f1_score(labels, predicted, average='weighted'))
            for i in range(num_of_classes):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(label_names().__len__()):
        label = label_names()[i]
        print("Size of %5s : %2d" % (label, counters_dev[label]))
        print('Accuracy of %5s : %2d %%' % (label, 100 * class_correct[i] / class_total[i]))

    print("F1-Score: {}.\t Recall: {}.\t Precision: {}.".format(mean(f1_scores), mean(recalls), mean(precisions)))
    print("End Evaluate Model")
    print("==============================================================================")
# End function


def train_model(model, data_loader, parameters):
    model_states = []
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=parameters['lr'], betas=parameters['betas'])

    for epoch in range(parameters['epochs']):
        running_loss = 0.0
        for idx, data in enumerate(data_loader, 0):
            model.train()

            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 50 == 49:
                if epoch + 1 >= 10:
                    model_states.append({
                        'state_dict': model.state_dict(),
                        'loss': running_loss / 50
                    })
                losses.append(running_loss / 50)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 50))
                running_loss = 0.0

    best_model = min(model_states, key=lambda x: x['loss'])
    torch.save({'model_state_dict': best_model}, './data/trained.ckpt')
    plot(losses, "Model Loss", "loss", "epoch")
    print("==============================================================================")
# End function


def create_data_loader(dataset, counters, parameters):
    labels = [label for _, label in dataset]
    class_weights = [dataset.__len__() / counters[label] for label in label_names().values()]
    weights = [class_weights[labels[i]] for i in range(dataset.__len__())]
    sampler = WeightedRandomSampler(weights=weights, num_samples=dataset.__len__())
    data_loader = DataLoader(dataset=dataset, batch_size=parameters['batch_size'], sampler=sampler)
    return data_loader
# End function


def plot(values, title, y_label, x_label):
    plt.plot(values)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(['val'], loc='upper left')
    plt.savefig('./{}.png'.format(title.replace(' ', '_')))
# End function


def main():
    # Create params dict for learning process
    parameters = {
        'num_classes': label_names().__len__(),
        'batch_size': 25,
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'epochs': 15
    }

    train_dataset = get_dataset_as_torch_dataset(path='./data/train.pickle')
    dev_dataset = get_dataset_as_torch_dataset(path='./data/dev.pickle')

    # Inspect dataset:
    counters_train, counters_dev = inspect_dataset(train_dataset, dev_dataset)

    # Load model
    model = SimpleModel()
    model.load(path='./data/pre_trained.ckpt')

    # Create Data Loaders for Train Dataset and Dev Dataset
    train_loader = create_data_loader(train_dataset, counters_train, parameters)
    dev_loader = create_data_loader(train_dataset, counters_dev, parameters)

    # Evaluate the given model
    evaluate_model(model, dev_loader, counters_dev)
    # Train the given model
    train_model(model, train_loader, parameters)
    # model.load(path='./data/trained.ckpt')
    evaluate_model(model, dev_loader, counters_dev)
# End function


if __name__ == '__main__':
    main()
