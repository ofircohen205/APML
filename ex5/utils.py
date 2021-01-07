import numpy as np
import torch
from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from matplotlib import pyplot as plt
import sys
import argparse


def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--name', type=str, required=True, help='the name of this run')
    p.add_argument('--log_dir', type=str, required=True, help='directory for tensorboard logs (common to many runs)')

    p.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
    p.add_argument('--num_samples', type=int, default=2000, help='number of samples from dataset')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--content_dim', type=int, default=32)
    p.add_argument('--class_dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--stddev', type=float, default=.3)

    args = p.parse_args()
    return args


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
# End function
