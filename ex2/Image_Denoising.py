import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
# from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW
from utils import *


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
        plt.subplot(2, 2, i + 1)
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
            image -= np.tile([dataset_mean_pixel], image.shape)
            # image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
            #                           image.shape[1])

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
    return X - np.tile(logsumexp(X, axis=0), (h, 1))
    # return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


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
    noisy_mses = {}
    denoised_mses = {}
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        noisy_mse = np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2)
        noisy_mses[str(noise_range[i])] = noisy_mse
        print(noisy_mse)
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        denoised_mse = np.mean((cropped_original - denoised_images[i]) ** 2)
        denoised_mses[str(noise_range[i])] = denoised_mse
        print(denoised_mse)

    plt.figure(figsize=(20, 20))
    plt.axis('off')
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1, xlabel='Noisy image', xticks=[], yticks=[])
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range), xlabel='Denoised image', xticks=[], yticks=[])
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()
    return noisy_mses, denoised_mses


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

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    D, M = X.shape
    X_normalized = normalize_log_likelihoods(X.copy())
    mvn = multivariate_normal(mean=model.mean, cov=model.cov)
    return mvn.logpdf(X_normalized.T).sum()
    # log_2pi = D * np.log(2 * np.pi)
    # log_det = np.log(np.linalg.det(model.cov))
    # residuals = calc_residuals(X_normalized, model.mean, "minus")
    # mahalanobis_distance = np.dot(np.dot(residuals.T, np.linalg.inv(model.cov)), residuals)
    # return -0.5 * (log_2pi + log_det + mahalanobis_distance).sum()


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    D, M = X.shape
    k = model.mix.shape[0]
    log_likelihood = 0
    for i in range(M):
        logpdf_X = 0
        for j in range(k):
            mvn = multivariate_normal(cov=model.cov[j, :])
            logpdf_X = mvn.logpdf(x=X[:, i]) * model.mix[j]
        log_likelihood += logpdf_X
    return log_likelihood


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


@benchmark
def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    return MVN_Model(np.mean(X, axis=-1), np.cov(X))


@benchmark
def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    D, M = X.shape
    rs = np.random.uniform(1, 1.1, k)
    cov = np.array([(rs[idx] * np.cov(X) + 1e-6 * np.identity(D)) for idx in range(k)])
    mix = np.random.rand(k)
    mix /= np.sum(mix)
    gsm_model = GSM_Model(cov, mix)
    return fit(X, gsm_model, rs)


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


@benchmark
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
    return calc_weiner_filter(Y, mvn_model.mean, mvn_model.cov, noise_std)


@benchmark
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
    X = np.empty(Y.shape)
    k = gsm_model.mix.shape[0]
    I = np.identity(gsm_model.cov[0, :].shape[0])
    for i in range(k):
        mvn = multivariate_normal(cov=(gsm_model.cov[i, :] + ((noise_std**2) * I)))
        upper_arg = gsm_model.mix[i] * (mvn.logpdf(Y[:, i]))
        lower_arg = 0
        for j in range(k):
            inner_mvn = multivariate_normal(cov=(gsm_model.cov[j] + ((noise_std**2) * I)))
            lower_arg += gsm_model.mix[j] * (inner_mvn.logpdf(Y[:, i]))
        c_i = upper_arg / lower_arg
        weiner_i = calc_weiner_filter(Y, np.zeros(Y.shape[0]), gsm_model.cov[i, :], noise_std)
        X += c_i * weiner_i
    return X


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


def fit(X, model, rs):
    D, M = X.shape
    k = model.mix.shape[0]
    lls = []
    ll = 1
    previous_ll = 0
    tol = 1e-2
    iter = 0
    cs = np.asmatrix(np.zeros((M, k), dtype=float))
    curr_time = current_time()
    while np.abs(ll - previous_ll) > tol:
        previous_ll = GSM_log_likelihood(X, model)
        EM(X, model, cs, rs)
        save_model(model, './output/gsm/gsm_model_{}_{}.pkl'.format(curr_time, iter))
        iter += 1
        ll = GSM_log_likelihood(X, model)
        print("EM Iteration no.: {}. GSM Log likelihood: {}".format(iter, ll))
        lls.append(ll)
    print("EM Done in iteration no.: {}. GSM Log likelihood: {}".format(iter, ll))
    # plot(lls, './output/gsm/plots')
    return model


def EM(X, model, cs, rs):
    # E-Step
    E_Step(X, model, cs)
    # M-Step
    M_Step(X, model, cs, rs)
# End function


def E_Step(X, model, cs):
    print("Start E-Step")
    D, M = X.shape
    k = model.mix.shape[0]
    for i in range(D):
        density = 0
        for j in range(k):
            mvn = multivariate_normal(cov=model.cov[j, :])
            c_i_j = mvn.logpdf(x=X[:, i]) * model.mix[j]
            density += c_i_j
            cs[i, j] = c_i_j
        cs[i, :] /= density
    cs = normalize_log_likelihoods(cs)
    print("End E-Step")
# End function


def M_Step(X, model, cs, rs):
    print("Start M-Step")
    D, M = X.shape
    k = model.mix.shape[0]
    for j in range(k):
        c_j = cs[:, j].sum()
        model.mix[j] = (1 / D) * c_j
        for i in range(D):
            rs[j] += np.sqrt((cs[i, j] * np.dot(np.dot(X[:, i].T, np.linalg.inv(model.cov[j, :])), X[:, i])))

        rs[j] = rs[j] / np.sqrt(D * c_j)
        # print("r_{}: {}".format(j, rs[j]))
        model.cov[j, :] = (rs[j] ** 2) * model.cov[j, :]
    print("End M-Step")
# End function


def calc_weiner_filter(Y, mean, cov, noise_std):
    cov_inv = np.linalg.inv(cov)
    eye_noise_std = np.eye(cov_inv.shape[0]) / (noise_std ** 2)
    left_arg = np.linalg.inv(cov_inv + eye_noise_std)
    right_arg = calc_residuals((Y / (noise_std ** 2)), np.dot(cov_inv, mean), "plus")
    X = np.dot(left_arg, right_arg)
    return X
# End function


if __name__ == '__main__':
    pass
