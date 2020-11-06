# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import *
from utils import *
from evaluator import *
from trainer import *
import torchvision.transforms as transforms
import torch.nn as nn
import os


##################
###### MAIN ######
##################
def main():
    if os.path.exists('./models') is not True:
        os.mkdir('./models/')
    if os.path.exists('./graphs') is not True:
        os.mkdir('./graphs/')
    train_dataset = get_dataset_as_torch_dataset(path='./data/train.pickle')
    dev_dataset = get_dataset_as_torch_dataset(path='./data/dev.pickle')

    # Create params dict for learning process
    parameters = {
        'train_size': train_dataset.__len__(),
        'dev_size': dev_dataset.__len__(),
        'num_classes': label_names().__len__(),
        'batch_size': 15,
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'epochs': 20,
        'criterion': nn.CrossEntropyLoss(),
        'epsilon': 0.15,
        'lrs': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]
    }

    # Inspect dataset:
    counters_train, counters_dev = inspect_dataset(train_dataset, dev_dataset)

    # Load model
    model = SimpleModel()
    model.load(path='./data/pre_trained.ckpt')

    # Create Data Loaders for Train Dataset and Dev Dataset
    # train_dataset = augment_dataset(train_dataset)
    train_loader = create_data_loader(train_dataset, counters_train, parameters, True)
    dev_loader = create_data_loader(dev_dataset, counters_dev, parameters, False)

    # Evaluate the given model on dev dataset
    evaluator = Evaluator(model, dev_loader, parameters['criterion'],
                          counters_dev, parameters['num_classes'], "evaluate_pre_train")
    evaluator.__evaluate__()

    # Train the given model and evaluate it
    trainer = Trainer(model, train_loader, parameters['criterion'], parameters['lr'],
                      parameters['betas'], parameters['epochs'], parameters['batch_size'],
                      parameters['num_classes'], parameters['epsilon'], "trainer")
    trainer.__train__(train_loader, True)

    evaluator = Evaluator(model, dev_loader, parameters['criterion'],
                          counters_dev, parameters['num_classes'], "evaluate_trainer")
    evaluator.__evaluate__()

    # # Model improvements
    # trainer = Trainer(model, dev_loader, parameters['criterion'], parameters['lr'],
    #                   parameters['betas'], 1, parameters['batch_size'],
    #                   parameters['num_classes'], "improved_trainer")
    # trainer.__train__(train_loader, False)
    # mislabeled = trainer.mislabeled
    # print(mislabeled)
    #
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# End function


if __name__ == '__main__':
    main()
