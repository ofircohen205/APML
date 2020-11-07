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
torch.manual_seed(17)


##################
###### MAIN ######
##################
def main():
    torch_train_dataset, torch_dev_dataset, parameters = init()

    # Load model
    pre_trained = SimpleModel()
    pre_trained.load(path=parameters['pretrained_path'])

    # Question 1
    # Inspect dataset:
    counters_train, counters_dev = inspect_dataset(torch_train_dataset, torch_dev_dataset)

    # Create Data Loaders for Train Dataset and Dev Dataset
    train_loader = create_data_loader(torch_train_dataset, counters_train, parameters, True)
    dev_loader = create_data_loader(torch_dev_dataset, counters_dev, parameters, False)

    evaluate(pre_trained, dev_loader, parameters, counters_dev, "pre_trained_eval", True)
    trainer = train(pre_trained, train_loader, parameters, "pre_trained_train", True)

    # Load the best state of the model we've trained
    trained = SimpleModel()
    trained.load(path=trainer.ckpt)
    evaluate(trained, dev_loader, parameters, counters_dev, "trained_eval", True)

    # Model improvements - get mislabeled images
    print("Improving given model:")
    mislabeled_loader = DataLoader(dataset=torch_train_dataset)
    improved_model = SimpleModel()
    improved_model.load(path=trainer.ckpt)
    evaluator = evaluate(improved_model, mislabeled_loader, parameters, counters_dev, "improved_model_eval", False)
    fix_dataset(evaluator, torch_train_dataset, parameters['fixed_dataset'])

    torch_train_dataset_fixed = get_dataset_as_torch_dataset(path=parameters['fixed_dataset'])
    train_loader = create_data_loader(torch_train_dataset_fixed, counters_train, parameters, True)
    counters_train, counters_dev = inspect_dataset(torch_train_dataset_fixed, torch_dev_dataset)

    test_model = SimpleModel()
    test_model.load(path=parameters['pretrained_path'])
    trainer = train(test_model, train_loader, parameters, "train_improved_model_train", True)
    eval_model = SimpleModel()
    eval_model.load(path=trainer.ckpt)
    evaluate(eval_model, dev_loader, parameters, counters_dev, "improved_model_eval", True)

    # # Question 2 - Playing with learning rate
    # models = [SimpleModel().load(trainer.ckpt) for _ in range(len(parameters['lrs']))]
    # for lr in parameters['lrs']:
    #     trainer = Trainer(model, train_loader, parameters['criterion'], parameters['lr'], parameters['betas'],
    #                       parameters['epochs'], parameters['batch_size'], parameters['num_classes'],
    #                       parameters['epsilon'], "trainer", parameters['path'])
    #     trainer.__train__(train_loader, True)

    # Question 3 - Adversarial example
    data_loader = DataLoader(dataset=torch_dev_dataset, batch_size=1)
    adversarial_model = SimpleModel()
    adversarial_model.load(path=trainer.ckpt)
    adversarial = Adversarial(adversarial_model, data_loader, parameters['adversarial_epsilons'], parameters['path_plots_adversarial'])
    adversarial.__attack__()
    adversarial.__plot_attack__()
    adversarial.__plot_examples__()


# End function


if __name__ == '__main__':
    main()
