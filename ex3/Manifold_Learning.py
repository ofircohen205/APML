import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from utils import *
from NetflixPreprocessing import *
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
    N, _ = X.shape
    H = np.eye(N) - np.ones((N, N)) / N
    S = -0.5 * np.dot(np.dot(H, (X**2)), H)

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(S)

    # sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Scree plot
    scree_plot(eigvals, "MDS", './plots/scree_plot_mds.png')

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
    X_distances = calculate_distances(X)
    N, _ = X_distances.shape

    # Create kernel similiary matrix
    kernel_similarity = np.exp(-X_distances**2 / sigma)

    # Normalize rows of K to form Markov Transition Matrix
    markov_transition_matrix = kernel_similarity / kernel_similarity.sum(axis=1).reshape(kernel_similarity.shape[1], 1)

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(markov_transition_matrix)

    # sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Scree plot
    scree_plot(eigvals, "Diffusion Maps", './plots/scree_plot_diffusion_maps.png')

    # Generate Nxd reduced dataset
    eigvals = eigvals
    diag_eigvals = np.asarray(np.diag(eigvals[1:(d+1)]**t))
    V = eigvecs[:, 1:(d+1)]
    result = np.dot(V, diag_eigvals)
    return result


def swiss_roll():
    from sklearn import manifold
    swiss_roll_dataset, color = datasets.make_swiss_roll(n_samples=2000)
    swiss_roll_dataset_distances = calculate_distances(swiss_roll_dataset)

    swiss_roll_dataset_mds = MDS(swiss_roll_dataset_distances, 2)
    plot_data(swiss_roll_dataset, swiss_roll_dataset_mds, color, './plots/plot_data_mds.png')

    # For checking my implementation - compared with sklearn results
    # swiss_roll_dataset_mds_sklearn = manifold.MDS(n_components=2).fit(swiss_roll_dataset_distances)
    # plot_data(swiss_roll_dataset, swiss_roll_dataset_mds_sklearn, color, './plots/plot_data_mds_sklearn.png')

    swiss_roll_dataset_diffusion_map = DiffusionMap(swiss_roll_dataset, 2, 100, 1000)
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

    # faces_diffusion_map = DiffusionMap(faces_dataset, 8, 100, 1000)
    # faces_fig_diffusion_map = plot_with_images(faces_diffusion_map, faces_dataset, "Diffusion Map Faces dataset")

    # For checking my implementation - compared with pydiffmap results
    # faces_fig_diffusion_map_pydiffmap = plot_with_images(faces_diffusion_map_pydiffmap, faces_dataset, "Diffusion Map Faces dataset pydiffmap")
    # faces_diffusion_map_pydiffmap = pydiffmap.diffusion_map.DiffusionMap(pydiffmap.kernel.Kernel(), n_evecs=8, alpha=0.5, ).fit_transform(faces_dataset)
# End function


def netflix():
    from sklearn.decomposition import NMF
    from sklearn.cluster import Birch
    mat_of_movies_and_users, df_of_movies_info = create_initial_data(remove_empty_cols=True, use_genres=True)
    nmf = NMF(n_components=1000, init='random', random_state=42).fit(mat_of_movies_and_users)
    plot_with_images(nmf, mat_of_movies_and_users, "Non negative matrix factorization")
    clustered_dataset_labels = Birch(n_clusters=1000).fit_predict(nmf)
# End function


def genetic():
    from sklearn import manifold
    genetic_dataset = load_dataset('./genetic_data.pickle')
    genetic_distances = calculate_distances(genetic_dataset)
    print("Done calculating distances - genetic dataset")
    genetic_dataset_mds = MDS(genetic_distances, 2)
    # genetic_dataset_mds_sklearn = manifold.MDS(n_components=2).fit(genetic_distances)
    plot_with_images(genetic_dataset_mds, genetic_dataset, "Genetic dataset")
    # plot_with_images(genetic_dataset_mds_sklearn, genetic_dataset, "Genetic dataset sklearn")
# End function


def main():
    create_dirs()
    # Swiss roll dataset
    # swiss_roll()

    # Faces dataset
    # faces()

    # Netflix Prize dataset
    netflix()
# End function


if __name__ == '__main__':
    main()
