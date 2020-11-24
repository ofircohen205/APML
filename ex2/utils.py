# Name: Ofir Cohen
# ID: 312255847
# Date: 25/11/2020
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
# from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW
import os, time
from datetime import datetime


def benchmark(func):
    def function_timer(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "Runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__, time=runtime))
        return value
    return function_timer
# End function


def plot(values, path):
    title = "GSM Log likelihood"
    plt.plot(values)
    plt.title(title)
    plt.ylabel("Log likelihood")
    plt.xlabel("Iterations")
    plt.legend(['val'], loc='upper left')
    fig_name = '{}/{}.png'.format(path, title.lower().replace(' ', '_'))
    plt.savefig(fig_name)
    plt.clf()
# End function


def load_dataset(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
# End function


def calc_residuals(X, mean, opr):
    residuals = X.copy()
    for idx in range(X.shape[0]):
        if opr == 'minus':
            residuals[idx] -= mean[idx]
        elif opr == 'plus':
            residuals[idx] += mean[idx]
    return residuals
# End function


def calculate_log_likelihood(model, residuals):
    return -0.5 * (np.log(np.linalg.det(model.cov))
                   + residuals.T.dot(np.linalg.inv(model.cov)).dot(residuals)
                   + 2 * np.log(2 * np.pi))
# End function


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
# End function


def create_dirs():
    if os.path.exists('./output') is not True:
        os.mkdir('./output/')
        os.mkdir('./output/mvn/')
        os.mkdir('./output/mvn/plots/')
        os.mkdir('./output/gsm/')
        os.mkdir('./output/gsm/plots/')
        os.mkdir('./output/ica/')
        os.mkdir('./output/ica/plots/')
# End function


def current_time():
    return datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
# End function
