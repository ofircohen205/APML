# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from evaluator import Evaluator
from trainer import Trainer
from adversarial import Adversarial
from utils import *
import torch


##################
###### MAIN ######
##################
def full_process(torch_train_dataset, torch_dev_dataset, train_loader, dev_loader, counters_train, counters_dev, parameters):
    # Load model
    pre_trained = SimpleModel()
    pre_trained.load(path=parameters['pretrained_path'])

    # Evaluate given model and train it
    evaluation(pre_trained, dev_loader, parameters, counters_dev, "pretrain_model", True)
    trainer = training_loop(pre_trained, train_loader, parameters, "pretrain_model")

    # Load the best state of the model we've trained and evaluate
    trained = SimpleModel()
    trained.load(path=trainer.ckpt)
    evaluation(trained, dev_loader, parameters, counters_dev, "trained_model", True)

    # Model improvements - get mislabeled images and fix them
    print("Improving given model:")
    print("Fix train dataset")
    torch_train_dataset_fixed = dataset_fix(torch_train_dataset, trainer, parameters, counters_dev, False, "train")
    train_loader = create_data_loader(torch_train_dataset_fixed, counters_train, parameters, True)
    print("==============================================================================")
    print("Fix dev dataset")
    # torch_dev_dataset_fixed = dataset_fix(torch_dev_dataset, trainer, parameters, counters_dev, False, "dev")
    # dev_loader = create_data_loader(torch_dev_dataset_fixed, counters_train, parameters, True)
    print("==============================================================================")

    counters_train, counters_dev = inspect_dataset(torch_train_dataset_fixed, torch_dev_dataset)

    # Test given model after fixing datasets
    print("Test given model after fixing datasets")
    test_model = SimpleModel()
    test_model.load(path=parameters['pretrained_path'])
    trainer = training_loop(test_model, train_loader, parameters, "train_improved_model_train")
    ckpt = trainer.ckpt
    eval_model = SimpleModel()
    eval_model.load(path=ckpt)
    evaluation(eval_model, dev_loader, parameters, counters_dev, "improved_model_eval", True)
    print("End part 1")
    print("==============================================================================")
    playing_with_learning_rate(train_loader, parameters)
    adversarial_example(torch_dev_dataset, parameters, ckpt)
    return ckpt
# End function


def evaluation(model, data_loader, parameters, counters, name, is_evaluating):
    evaluate(model, data_loader, parameters, counters, name, is_evaluating)
# End function


def training_loop(model, data_loader, parameters, name):
    return train(model, data_loader, parameters, name, True)
# End function


def train_and_eval(dev_loader, train_loader, counters, parameters, name, is_evaluating, path):
    model = SimpleModel()
    model.load(path)
    evaluate(model, dev_loader, counters, parameters, name, is_evaluating)
    return training_loop(model, train_loader, parameters, name)
# End function


def dataset_fix(dataset, trainer, parameters, counters_dev, is_evaluating, name):
    mislabeled_train_loader = DataLoader(dataset=dataset)
    improved_train_model = SimpleModel()
    improved_train_model.load(path=trainer.ckpt)
    evaluator = evaluate(improved_train_model, mislabeled_train_loader, parameters, counters_dev, "improved_model_eval",
                         is_evaluating)
    if name == "train":
        output = parameters['fixed_dataset']
    else:
        output = parameters['fixed_dataset_dev']
    fix_dataset(evaluator, dataset, output)
    torch_dataset_fixed = get_dataset_as_torch_dataset(path=output)
    return torch_dataset_fixed
# End function


def playing_with_learning_rate(train_loader, parameters):
    print("Start part 2")
    # Question 2 - Playing with learning rate
    print("Playing with learning rate")
    models = [SimpleModel() for _ in range(len(parameters['lrs']))]
    for idx in range(len(parameters['lrs'])):
        models[idx].load(parameters['pretrained_path'])
        model_name = "model_{}".format(idx)
        lr = parameters['lrs'][idx]
        print("model: {}. lr: {}".format(model_name, lr))
        trainer = Trainer(models[idx], train_loader, parameters['criterion'], lr,
                          parameters['betas'], parameters['epochs'], parameters['batch_size'],
                          parameters['num_classes'], parameters['epsilon'], model_name, parameters['path_lrs'])
        trainer.__train__(False)
        plot(trainer.losses, "{} Loss".format(trainer.name), "loss", "epoch", parameters['path_lrs'])
        plot(trainer.accuracies, "{} Accuracy".format(trainer.name), "accuracy", "epoch", parameters['path_lrs'])
    print("End Playing with learning rate")
    print("End part 2")
# End function


def adversarial_example(torch_dev_dataset, parameters, ckpt):
    print("Start part 3")
    # Question 3 - Adversarial example
    data_loader = DataLoader(dataset=torch_dev_dataset, batch_size=1)
    adversarial_model = SimpleModel()
    adversarial_model.load(path=ckpt)
    adversarial = Adversarial(adversarial_model, data_loader, parameters['adversarial_epsilons'],
                              parameters['path_plots_adversarial'])
    adversarial.__attack__()
    adversarial.__plot_attack__()
    adversarial.__plot_examples__()
    print("End part 3")
# End function


def main():
    torch_train_dataset, torch_dev_dataset, parameters = init()

    # Inspect dataset:
    counters_train, counters_dev = inspect_dataset(torch_train_dataset, torch_dev_dataset)

    # Create Data Loaders for Train Dataset and Dev Dataset
    train_loader = create_data_loader(torch_train_dataset, counters_train, parameters, True)
    dev_loader = create_data_loader(torch_dev_dataset, counters_dev, parameters, False)

    # If you wish not to executed all training life cycle - you can comment the full_process function
    # and uncomment the wanted function
    ckpt = full_process(torch_train_dataset, torch_dev_dataset, train_loader, dev_loader, counters_train, counters_dev, parameters)

    # path = 'YOUR_PATH'
    # model = SimpleModel()
    # model.load(path)
    # evaluation(model, dev_loader, parameters, counters_dev, "eval", True)

    # trainer = train_and_eval(train_loader, dev_loader, counters_dev, parameters, "improved_model", True, path)
    # ckpt = trainer.ckpt

    # playing_with_learning_rate(train_loader, parameters)
    # adversarial_example(torch_dev_dataset, parameters, ckpt)


# End function


if __name__ == '__main__':
    main()
