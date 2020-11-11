# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from evaluator import Evaluator
from trainer import Trainer
from dataset import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


#######################
###### FUNCTIONS ######
#######################
def init():
    # Create dir for results
    output_dir = create_dirs_if_needed()

    # Create params dict for learning process
    parameters = {
        'train_dataset_path': './data/train.pickle',
        'dev_dataset_path': './output/fixed_dev.pickle',
        'num_classes': label_names().__len__(),
        'batch_size': 15,
        'lr': 1e-2,
        'betas': (0.9, 0.999),
        'epochs': 30,
        'criterion': nn.CrossEntropyLoss(),
        'adversarial_epsilons': [0, .05, .1, .15, .2, .25, .3],
        'epsilon': 0.15,
        'lrs': [1e-4, 1e-2, 1, 1e+2, 1e+4],
        'path': output_dir,
        'path_lrs': os.path.join(output_dir, "lrs/"),
        'path_conf_matrix': os.path.join(output_dir, "confusion_matrix/"),
        'path_plots_nn': os.path.join(output_dir, "plots/training_nn/"),
        'path_plots_adversarial': os.path.join(output_dir, "plots/adversarial/"),
        'pretrained_path': './data/pre_trained.ckpt',
        'fixed_dataset': './output/fixed_train_dataset.pickle',
        'fixed_dataset_dev': './output/fixed_dev_dataset.pickle'
    }

    # Create objects that holds datasets
    torch_train_dataset = get_dataset_as_torch_dataset(path=parameters['train_dataset_path'])
    torch_dev_dataset = get_dataset_as_torch_dataset(path=parameters['dev_dataset_path'])
    parameters['train_size'] = torch_train_dataset.__len__()
    parameters['dev_size'] = torch_dev_dataset.__len__()

    return torch_train_dataset, torch_dev_dataset, parameters
# End function


def evaluate(model, data_loader, parameters, counters_dev, name, create_conf_matrix):
    # Evaluate the given model on dev dataset
    print("Start Evaluate Model")
    evaluator = Evaluator(model, data_loader, parameters['criterion'],
                          counters_dev, parameters['num_classes'], name)
    if create_conf_matrix:
        evaluator.__evaluate__()
        conf_mat = confusion_matrix(evaluator.labels.numpy(), evaluator.predicted.numpy())
        f_score = f1_score(evaluator.labels.numpy(), evaluator.predicted.numpy(), average='macro')
        labels_names = label_names().values()
        plot_confusion_matrix_sns(conf_mat, "{} confusion matrix".format(evaluator.name),
                                  labels_names, parameters['path_conf_matrix'])
        print("Model {}. F1-Score: {}".format(evaluator.name, f_score))
    else:
        evaluator.__get_mislabeled__()

    print("End Evaluate Model")
    print("==============================================================================")
    return evaluator
# End function


def train(model, data_loader, parameters, name, save_best_ckpt):
    # Train the given model and evaluate it
    print("Start Train Model")
    trainer = Trainer(model, data_loader, parameters['criterion'], parameters['lr'], parameters['betas'],
                      parameters['epochs'], parameters['batch_size'], parameters['num_classes'],
                      parameters['epsilon'], name, parameters['path'])
    trainer.__train__(save_best_ckpt)
    if save_best_ckpt:
        # if len(trainer.model_states) > 0:
        #     trainer.model_states = sorted(trainer.model_states, key=lambda x: x['loss'])
        #     torch.save({'model_state_dict': trainer.model_states[0]['state_dict']}, trainer.ckpt)
        # else:
        trainer.model.save(trainer.ckpt)

    plot(trainer.losses, "{} Loss".format(trainer.name), "loss", "epoch", parameters['path_plots_nn'])
    plot(trainer.accuracies, "{} Accuracy".format(trainer.name), "accuracy", "epoch", parameters['path_plots_nn'])
    print("End Train Model")
    print("==============================================================================")

    return trainer
# End function


def inspect_dataset(train_dataset, dev_dataset):
    """
    :param train_dataset:
    :param dev_dataset:
    :return counters_train, counters_dev:
    """
    print("Start Inspecting Dataset")
    train_size = train_dataset.__len__()
    dev_size = dev_dataset.__len__()
    print("Number of train images: {}. Number of dev images: {}".format(train_size, dev_size))
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
    train_str = ""
    for cls in counters_train:
        print("Class {}. size: {}. Percentage: {}".format(cls, counters_train[cls], (counters_train[cls] / train_size)))

    print("Dev dataset:")
    for cls in counters_dev:
        print("Class {}. size: {}. Percentage: {}".format(cls, counters_dev[cls], (counters_dev[cls] / dev_size)))

    print("End Inspecting Dataset")
    print("==============================================================================")
    return counters_train, counters_dev
# End function


def create_data_loader(dataset, counters, parameters, init_sampler):
    labels = [label for _, label in dataset.the_list]
    class_weights = [dataset.__len__() / counters[label] for label in label_names().values()]
    weights = [class_weights[labels[i]] for i in range(dataset.__len__())]
    if init_sampler:
        sampler = WeightedRandomSampler(weights=weights, num_samples=dataset.__len__())
        data_loader = DataLoader(dataset=dataset, batch_size=parameters['batch_size'], sampler=sampler)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=parameters['batch_size'])
    return data_loader
# End function


def plot(values, title, y_label, x_label, path):
    plt.plot(values)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(['val'], loc='upper left')
    fig_name = '{}/{}.png'.format(path, title.lower().replace(' ', '_'))
    plt.savefig(fig_name)
    plt.clf()
# End function


def plot_confusion_matrix_sns(values, title, classes, path):
    ax = sns.heatmap(values, xticklabels=classes, yticklabels=classes, annot=True,
                     cmap='coolwarm', linewidths=0.5, fmt='d')
    ax.set_ylim(3.0, 0)
    plt.title(title)
    fig_name = '{}/{}.png'.format(path, title.lower().replace(' ', '_'))
    plt.savefig(fig_name)
    plt.clf()
# End function


def create_dirs_if_needed():
    output_dir = './output/model_{}/'.format(current_time())
    if os.path.exists('./output') is not True:
        os.mkdir('./output/')
    if os.path.exists(output_dir) is not True:
        os.mkdir(output_dir)
        os.mkdir(output_dir + "lrs/")
        os.mkdir(output_dir + "confusion_matrix/")
        os.mkdir(output_dir + "plots/")
        os.mkdir(output_dir + "plots/training_nn/")
        os.mkdir(output_dir + "plots/adversarial/")

    return output_dir
# End function


def fix_dataset(evaluator, dataset, path):
    mislabeled = sorted(evaluator.mislabeled, key=lambda x: x['loss'], reverse=True)
    fixed_dataset = []
    for i in range(len(mislabeled)):
        inputs, labels = dataset.__getitem__(mislabeled[i]['index'])
        actual = mislabeled[i]['labels']
        predicted = mislabeled[i]['predicted']
        if label_names()[actual] == 'car' and label_names()[predicted] == 'cat':
            labels = predicted
            fixed_dataset.append((inputs, labels))
        else:
            fixed_dataset.append((inputs, labels))

    with open(path, 'wb') as f:
        pickle.dump(fixed_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
# End function


def current_time():
    return datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
# End function
