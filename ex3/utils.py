import pickle, os
import matplotlib.pyplot as plt
import numpy as np


def load_dataset(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
# End function


def calculate_distances(dataset):
    from sklearn.metrics.pairwise import euclidean_distances
    return euclidean_distances(X=dataset)
# End function


def generate_noised_dataset():
    pass
# End function


def scree_plot(eig_vals, name, file_name):
    plt.figure()
    xs = np.arange(len(eig_vals)) + 1
    plt.plot(xs, eig_vals, 'ro-', linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Matrix size")
    plt.ylabel("Eigenvalue")
    name = "Eigenvalues from {name}".format(name=name)
    plt.legend([name], loc='best')
    plt.savefig(file_name)
    plt.clf()
# End function


def plot_data(dataset, dataset_reduced, color, file_name):
    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    ax = fig.add_subplot(212)
    ax.scatter(dataset_reduced[:, 0], dataset_reduced[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title('Projected data')
    plt.savefig(file_name)
    plt.clf()
# End function


def create_dirs():
    if not os.path.exists('./plots/'):
        os.mkdir('./plots/')
    if not os.path.exists('./dataset/'):
        os.mkdir('./dataset/')
# End function
