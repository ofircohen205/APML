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
def main():
    torch_train_dataset, torch_dev_dataset, parameters = init()

    # Load model
    pre_trained = SimpleModel()
    pre_trained.load(path=parameters['pretrained_path'])

    # # Question 1
    # Inspect dataset:
    counters_train, counters_dev = inspect_dataset(torch_train_dataset, torch_dev_dataset)

    # Create Data Loaders for Train Dataset and Dev Dataset
    train_loader = create_data_loader(torch_train_dataset, counters_train, parameters, True)
    dev_loader = create_data_loader(torch_dev_dataset, counters_dev, parameters, False)

    # Evaluate given model and train it
    evaluate(pre_trained, dev_loader, parameters, counters_dev, "pre_trained_eval", True)
    trainer = train(pre_trained, train_loader, parameters, "pre_trained_train", True)

    # Load the best state of the model we've trained and evaluate
    trained = SimpleModel()
    trained.load(path=trainer.ckpt)
    evaluate(trained, dev_loader, parameters, counters_dev, "trained_eval", True)

    # Model improvements - get mislabeled images and fix them
    print("Improving given model:")

    print("Fix train dataset")
    mislabeled_train_loader = DataLoader(dataset=torch_train_dataset)
    improved_train_model = SimpleModel()
    improved_train_model.load(path=trainer.ckpt)
    evaluator = evaluate(improved_train_model, mislabeled_train_loader, parameters, counters_dev, "improved_model_eval", False)
    fix_dataset(evaluator, torch_train_dataset, './output/train_dataset.pickle')
    torch_train_dataset_fixed = get_dataset_as_torch_dataset(path='./output/train_dataset.pickle')
    train_loader = create_data_loader(torch_train_dataset_fixed, counters_train, parameters, True)
    print("==============================================================================")

    print("Fix dev dataset")
    mislabeled_dev_loader = DataLoader(dataset=torch_dev_dataset)
    improved_dev_model = SimpleModel()
    improved_dev_model.load(path=trainer.ckpt)
    evaluator = evaluate(improved_dev_model, mislabeled_dev_loader, parameters, counters_dev, "improved_model_eval", False)
    fix_dataset(evaluator, torch_dev_dataset, './output/dev_dataset.pickle')
    torch_dev_dataset_fixed = get_dataset_as_torch_dataset('./output/dev_dataset.pickle')
    dev_loader = create_data_loader(torch_dev_dataset_fixed, counters_train, parameters, True)
    print("==============================================================================")

    counters_train, counters_dev = inspect_dataset(torch_train_dataset_fixed, torch_dev_dataset_fixed)

    # Test given model after fixing datasets
    print("Test given model after fixing datasets")
    test_model = SimpleModel()
    test_model.load(path=parameters['pretrained_path'])
    trainer = train(test_model, train_loader, parameters, "train_improved_model_train", True)
    ckpt = trainer.ckpt
    eval_model = SimpleModel()
    eval_model.load(path=ckpt)
    evaluate(eval_model, dev_loader, parameters, counters_dev, "improved_model_eval", True)
    print("End part 1")
    print("==============================================================================")

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
    print("Start part 3")
    # Question 3 - Adversarial example
    data_loader = DataLoader(dataset=torch_dev_dataset, batch_size=1)
    adversarial_model = SimpleModel()
    adversarial_model.load(path=ckpt)
    adversarial = Adversarial(adversarial_model, data_loader, parameters['adversarial_epsilons'], parameters['path_plots_adversarial'])
    adversarial.__attack__()
    adversarial.__plot_attack__()
    adversarial.__plot_examples__()
    print("End part 3")

# End function


if __name__ == '__main__':
    main()
