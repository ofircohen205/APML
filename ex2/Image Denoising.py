import time

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import pickle
from skimage.util import view_as_windows as viewW

def print_red(skk): print("\033[91m {}\033[00m" .format(skk))
def print_green(skk): print("\033[92m {}\033[00m" .format(skk))
def print_yellow(skk): print("\033[93m {}\033[00m" .format(skk))
def print_light_purple(skk): print("\033[94m {}\033[00m" .format(skk))
def print_purple(skk): print("\033[95m {}\033[00m" .format(skk))
def print_cyan(skk): print("\033[96m {}\033[00m" .format(skk))
def pr_light_gray(skk): print("\033[97m {}\033[00m" .format(skk))
def print_black(skk): print("\033[98m {}\033[00m" .format(skk))


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                        window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
        noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """
    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix

class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """
    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeX number_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    start = time.time()
    x_log = normalize_log_likelihoods(X.copy())
    ll = multivariate_normal.logpdf(x_log.T, model.mean, model.cov).sum()
    end = time.time()
    print_cyan(f'MVN_log_likelihood() ran for:{ end - start } seconds')
    return ll


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    print_green('--- GSM_log_likelihood() executed ---')
    m, n = X.shape
    ll = 0
    start = time.time()
    for i in range(n):
        tmp = 0
        for j in range(model.mix.shape[0]):
            tmp += multivariate_normal.logpdf(x=X[:, i], cov=model.cov[j, :]) * model.mix[j]
        ll += tmp
    end = time.time()
    print_cyan(f'GSM_log_likelihood() ran for:{ end - start } seconds')
    print_green('--- GSM_log_likelihood() done ---')
    return ll


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    print_green('--- learn_MVN() executed ---')
    mvn = MVN_Model(np.mean(X, axis=1), np.cov(X))
    print_green('--- learn_MVN() done ---')
    return mvn


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    print_green('--- learn_GSM() executed ---')
    m, n = X.shape
    cov = np.array([np.cov(X) for i in range(k)])
    _rYs = np.ones(k)
    phi = np.ones(k) / k
    model = GSM_Model(cov, phi)

    num_iterations = 0
    ll = 1
    previous_ll = 0
    tol = 1e-2
    w = np.asmatrix(np.empty((m, k), dtype=float))
    while abs(ll - previous_ll) > tol:
        previous_ll = GSM_log_likelihood(X, model)
        fit_GSM(model, X, w, k, _rYs)
        num_iterations += 1
        ll = GSM_log_likelihood(X, model)
        print_purple('Iteration %d: log-likelihood is %.6f' % (num_iterations, ll))
        print_light_purple(f'current log likelihood - previous likelihood = {ll - previous_ll}')
        save_model(model=model)
    print_yellow('Terminate at %d-th iteration:log-likelihood is %.6f' % (num_iterations, ll))
    print_green('--- learn_GSM() done ---')
    return model


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    # TODO: YOUR CODE HERE


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    # print(mvn_model.mean.shape)
    print_green('--- MVN_Denoise() executed ---')
    arg_1 = np.linalg.inv(np.linalg.inv(mvn_model.cov) + (1/noise_std**2) * np.identity(mvn_model.cov.shape[0]))

    arg_2_1 = np.dot(np.linalg.inv(mvn_model.cov), mvn_model.mean)
    arg_2 = (1/noise_std**2) * Y

    for i in range(arg_2.shape[0]):
        arg_2[i] += arg_2_1[i]

    print_green('--- MVN_Denoise() done ---')
    return np.dot(arg_1, arg_2)


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """

    # TODO: YOUR CODE HERE


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    # TODO: YOUR CODE HERE

def load_data(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def do_mvn_stuff(train_data, train_samples, test_data, test_samples):
    mvn_model = learn_MVN(train_samples)
    ll = MVN_log_likelihood(train_samples, mvn_model)
    for img in grayscale_and_standardize(train_data):
        test_denoising(img, mvn_model, MVN_Denoise)
    print(f'Mvn log likelihood estimation: {ll}')


def do_gsm_stuff(train_data, train_samples, test_data, test_samples):
    gsm_model = learn_GSM(train_samples, 10)

def fit_GSM(gsm_model, X, w, k, _rYs):
    """
    Implementation for EM algorithm.

    :param gsm_model: pre-trained GSM model.
    :param X: The data object.
    :param w: ciy -> the probability for Xi sample to be drawn from gaussian Yi.
    :param k: number of gaussian's.
    :param _rYs the scalars we are train for each model
    """
    print_green('--- fit_GSM() executed ---')
    gsm_e_step(gsm_model, X, w, k)
    gsm_m_step(gsm_model, X, w, k, _rYs)
    print_green('--- fit_GSM() done ---')

def gsm_e_step(gsm_model, X, w, k):
    print_red('--- gsm_e_step() executed ---')
    m, n = X.shape
    for i in range(m):
        den = 0
        for j in range(k):
            num = multivariate_normal.pdf(x=X[:, i], cov=gsm_model.cov[j, :]) * gsm_model.mix[j]
            den += num
            w[i, j] = num
        w[i, :] /= den
        # assert w[i, :].sum() - 1 < 1e-4
    print_red('--- gsm_e_step() done ---')

def gsm_m_step(gsm_model, X, w, k, _rYs):
    print_red('--- gsm_m_step() executed ---')
    m, n = X.shape
    for j in range(k):
        const = w[:, j].sum()
        gsm_model.mix[j] = 1 / m * const
        _sigma_j = np.zeros((m, m))
        for i in range(m):
            _rYs[j] += np.sqrt((w[i, j] * np.dot(np.dot(X[:, i].T, np.linalg.inv(gsm_model.cov[j, :])), X[:, i])))
            _sigma_j += w[i, j] * (X[:, i].T * X[:, i])

        _rYs[j] = _rYs[j] / np.sqrt(m * const)
        print(_rYs[j])
        gsm_model.cov[j, :] = (_rYs[j]**2) * gsm_model.cov[j, :]
    print_red('--- gsm_m_step() done ---')

def save_model(model):
    import pickle
    from datetime import datetime
    with open(f'./models/GSM_{datetime.now().strftime("%d-%m-%Y_%H_%M_%S")}.pkl', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_data = load_data('train_images.pickle')
    test_data = load_data('test_images.pickle')

    train_samples = sample_patches(train_data)
    test_samples = sample_patches(test_data)

    # do_mvn_stuff(train_data, train_samples, test_data, test_samples)
    do_gsm_stuff(train_data, train_samples, test_data, test_samples)

