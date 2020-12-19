import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from utils import *
from Netflix_Challenge import *
from sklearn.manifold import TSNE
from sklearn.decomposition import SparsePCA
from mpl_toolkits.mplot3d import Axes3D
import pydiffmap


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()


def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''
    eigvals, eigvecs = get_mds_eig_entities(X)

    # Generate Nxd reduced dataset
    diag_eigvals = np.diag(np.sqrt(eigvals[:d]))
    V = eigvecs[:, :d]
    result = np.dot(V, diag_eigvals)
    return result


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''
    from sklearn.manifold import LocallyLinearEmbedding
    return LocallyLinearEmbedding(n_neighbors=k, n_components=d).fit_transform(X=X)


def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''
    eigvals, eigvecs = get_diffusion_maps_eig_entites(X, sigma)

    # Generate Nxd reduced dataset
    diag_eigvals = np.asarray(np.diag(eigvals[1:(d+1)]**t))
    V = eigvecs[:, 1:(d+1)]
    result = np.dot(V, diag_eigvals)
    return result


def swiss_roll():
    from sklearn import manifold
    swiss_roll_dataset, color = datasets.make_swiss_roll(n_samples=2000)
    swiss_roll_dataset_distances = calculate_distances(swiss_roll_dataset)

    swiss_roll_dataset_mds = MDS(swiss_roll_dataset_distances, 2)
    # Scree plot
    eigvals, _ = get_mds_eig_entities(swiss_roll_dataset_distances)
    scree_plot(eigvals, "MDS", './plots/swiss_roll_scree_plot_mds.png')

    plot_data(swiss_roll_dataset, swiss_roll_dataset_mds, color, './plots/plot_data_mds.png')

    # For checking my implementation - compared with sklearn results
    # swiss_roll_dataset_mds_sklearn = manifold.MDS(n_components=2).fit(swiss_roll_dataset_distances)
    # plot_data(swiss_roll_dataset, swiss_roll_dataset_mds_sklearn, color, './plots/plot_data_mds_sklearn.png')

    swiss_roll_dataset_diffusion_map = DiffusionMap(swiss_roll_dataset, 2, 100, 1000)
    # Scree plot
    eigvals, _ = get_diffusion_maps_eig_entites(swiss_roll_dataset, 100)
    scree_plot(eigvals, "Diffusion Maps", './plots/scree_plot_diffusion_maps.png')
    plot_data(swiss_roll_dataset, swiss_roll_dataset_diffusion_map, color, './plots/plot_data_diffusion_map.png')

    # For checking my implementation - compared with pydiffmap results
    # swiss_roll_dataset_diffusion_map_pydiffmap = pydiffmap.diffusion_map.DiffusionMap(pydiffmap.kernel.Kernel(), n_evecs=2).fit_transform(swiss_roll_dataset)
    # plot_data(swiss_roll_dataset, swiss_roll_dataset_diffusion_map_pydiffmap, color, './plots/plot_data_diffusion_map_pydiffmap.png')

    swiss_roll_dataset_lle = LLE(swiss_roll_dataset, 2, 12)
    plot_data(swiss_roll_dataset, swiss_roll_dataset_lle, color, './plots/plot_data_lle.png')
# End function


def faces():
    from sklearn import manifold
    faces_dataset = load_dataset('./faces.pickle')
    faces_distances = calculate_distances(faces_dataset)

    faces_mds = MDS(faces_distances, 8)
    faces_fig_mds = plot_with_images(faces_mds, faces_dataset, "MDS Faces dataset")

    # For checking my implementation - compared with sklearn results
    # faces_mds_sklearn = manifold.MDS(n_components=8).fit(faces_distances)
    # faces_fig_mds_sklearn = plot_with_images(faces_mds_sklearn, faces_dataset, "MDS Faces dataset sklearn")

    faces_lle = LLE(faces_dataset, 8, 12)
    faces_fig_lle = plot_with_images(faces_lle, faces_dataset, "LLE Faces dataset")

    faces_diffusion_map = DiffusionMap(faces_dataset, 8, 100, 1000)
    faces_fig_diffusion_map = plot_with_images(faces_diffusion_map, faces_dataset, "Diffusion Map Faces dataset")

    # For checking my implementation - compared with pydiffmap results
    # faces_fig_diffusion_map_pydiffmap = plot_with_images(faces_diffusion_map_pydiffmap, faces_dataset, "Diffusion Map Faces dataset pydiffmap")
    # faces_diffusion_map_pydiffmap = pydiffmap.diffusion_map.DiffusionMap(pydiffmap.kernel.Kernel(), n_evecs=8, alpha=0.5).fit_transform(faces_dataset)
# End function


def noised_dataset():
    num_samples = 2000
    orthogonal_matrix_size = 75
    noise = [0, 0.1, 0.5, 0.9, 1]
    colors = ['r', 'b', 'g', 'y', 'k']
    data, orthogonal_matrix = generate_noised_dataset()
    plt.figure(figsize=(8, 5))
    for i, n in enumerate(noise):
        noisy_data = np.dot(data, orthogonal_matrix[:2, :]) + np.random.normal(0, n, size=(num_samples, orthogonal_matrix_size))
        euclidean_dists = calculate_distances(noisy_data)
        eigvals, eigvecs = get_mds_eig_entities(euclidean_dists)
        scree_plot_noised(np.sort(eigvals)[::-1][:60], 60, colors[i], f'noise={n}', 'Eigenvalues', 'MDS')
        MDS(euclidean_dists, 2)
    plt.legend(loc='best')
    plt.show()
# End function


def netflix():
    netflix_challenge = Netflix_Challenge(True, True)
    # netflix_challenge.__inspect__movies__()
    # netflix_challenge.__inspect__rating__()
    techniques = ('TSNE', 'LLE')
    interval = [1995.0, 1996.0, 1997.0, 1998.0, 1999.0, 2000.0, 2001.0, 2002.0]

    # tsne = netflix_challenge.__execute_manifold__(technique=techniques.__getitem__(0), interval=interval)
    # netflix_challenge.__plot_scatter__(tsne, 'TSNE Scatter', '3D')

    # lle = netflix_challenge.__execute_manifold__(technique=techniques.__getitem__(1), interval=interval)
    # netflix_challenge.__plot_scatter__(lle, 'LLE Scatter', '2D')

    clusters, reduced_data = netflix_challenge.__execute_clustering__(technique=techniques.__getitem__(0), interval=interval)
    netflix_challenge.__plot_clustering__(clusters, reduced_data)
# End function


def main():
    create_dirs()
    # Swiss roll dataset
    # swiss_roll()

    # Faces dataset
    # faces()

    # Noised dataset
    # noised_dataset()

    # Netflix Prize dataset
    netflix()
# End function


if __name__ == '__main__':
    main()
