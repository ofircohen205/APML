# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from dataset import *
from utils import *
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F


#####################
###### CLASSES ######
#####################
class Adversarial:
    """
    Adversarial class for training a given model
    """

    def __init__(self, model: SimpleModel, dataset: DataLoader, epsilons: list, path: str):
        """
        :param model: the model we wish to evaluate
        :param dataset: the data we wish to evaluate our model on
        :param criterion: loss function
        :param epsilon:
        """
        self.model = model
        self.dataset = dataset
        self.epsilons = epsilons
        self.path = path
        self.accuracies = []
        self.examples = []

    def __attack__(self):
        # Run test for each epsilon
        for eps in self.epsilons:
            acc, ex = self.__test__(eps)
            self.accuracies.append(acc)
            self.examples.append(ex)

    def __test__(self, eps):
        # Accuracy counter
        correct = 0
        adv_examples = []

        # Loop over all examples in test set
        for inputs, labels in self.dataset:
            # Set requires_grad attribute of tensor. Important for Attack
            inputs.requires_grad = True

            # Forward pass the data through the model
            output = self.model(inputs)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != labels.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, labels)

            # Zero all existing gradients
            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = inputs.grad.data

            # Call FGSM Attack
            perturbed_data = self.__fsgm_attack__(inputs, eps, data_grad)

            # Re-classify the perturbed image
            output = self.model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == labels.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (eps == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        # Calculate final accuracy for this epsilon
        final_acc = correct / float(len(self.dataset))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(self.dataset), final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples

    def __fsgm_attack__(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    def __plot_attack__(self):
        plt.figure(figsize=(5, 5))
        plt.plot(self.epsilons, self.accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        fig_name = '{}/{}.png'.format(self.path, "adversarial_plot")
        plt.savefig(fig_name)
        plt.clf()

    def __plot_examples__(self):
        # Plot several examples of adversarial samples at each epsilon
        for i in range(len(self.epsilons)):
            for j in range(len(self.examples[i])):
                plt.xticks([], [])
                plt.yticks([], [])
                plt.ylabel("Eps: {}".format(self.epsilons[i]), fontsize=14)
                orig, adv, ex = self.examples[i][j]
                plt.title("Original: {} -> Predicted: {}".format(label_names()[orig], label_names()[adv]))
                plt.imshow(un_normalize_image(ex))
                example = "example_{}_{}".format(i, j)
                fig_name = '{}/{}.png'.format(self.path, example)
                plt.savefig(fig_name)
                plt.clf()
# End class
