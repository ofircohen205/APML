# Name: Ofir Cohen
# ID: 312255847

###################
##### IMPORTS #####
###################
from models import SimpleModel
from dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import *
import torch


#####################
###### CLASSES ######
#####################
class Evaluator:
    """
    Evaluator class for evaluating a given model
    """
    def __init__(self, model: SimpleModel, dataset: DataLoader, criterion, counters: dict, num_classes: int, name: str):
        """
        :param model: the model we wish to evaluate
        :param dataset: the data we wish to evaluate our model on
        :param counters: the number of data in each class
        :param num_classes: the number of classes
        """
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.counters = counters
        self.num_classes = num_classes
        self.name = name
        self.predicted = torch.zeros(0, dtype=torch.long)
        self.labels = torch.zeros(0, dtype=torch.long)
        self.mislabeled = []

    def __evaluate__(self):
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))

        self.model.eval()
        with torch.no_grad():
            for data in self.dataset:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels).squeeze()
                self.predicted = torch.cat([self.predicted, predicted.view(-1)])
                self.labels = torch.cat([self.labels, labels.view(-1)])

                for idx in range(self.num_classes):
                    label = labels[idx]
                    class_correct[label] += c[idx].item()
                    class_total[label] += 1

        for idx in range(self.num_classes):
            label = label_names()[idx]
            print("Size of %5s : %2d" % (label, self.counters[label]))
            print('Accuracy of %5s : %2d %%' % (label, 100 * class_correct[idx] / class_total[idx]))

        print('Overall Accuracy: %2d %%' % (100 * sum(class_correct) / sum(class_total)))

    def __get_mislabeled__(self):
        print("Start get mislabeled images")
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(self.dataset, 0):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                self.mislabeled.append({'index': idx,
                                        'labels': labels.item(),
                                        'predicted': predicted.item(),
                                        'loss': loss.item()})

        print("End get mislabeled images")
        print("==============================================================================")
# End class
