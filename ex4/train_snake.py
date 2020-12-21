import shutil
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import argparse
import os
import sys
import time

from q_policy import QPolicy
from snake_wrapper import SnakeWrapper
from models import SimpleModel, DqnModel, MonteCarloModel


def create_model(network_name):
    """
    Kind-of model factory.
    Edit it to add more models.
    :param network_name: The string input from the terminal
    :return: The model
    """
    if network_name == 'simple':
        return SimpleModel()
    elif network_name == 'dqn':
        return DqnModel()
    elif network_name == 'monte_carlo':
        return MonteCarloModel()
    else:
        raise Exception('net {} is not known'.format(network_name))


def create_policy(policy_name,
                  buffer_size, gamma, model: torch.nn.Module, writer: SummaryWriter, lr):
    """
    Kind-of policy factory.
    Edit it to add more algorithms.
    :param policy_name: The string input from the terminal
    :param buffer_size: size of policy's buffer
    :param gamma: reward decay factor
    :param model: the pytorch model
    :param writer: tensorboard summary writer
    :param lr: initial learning rate
    :return: A policy object
    """
    if policy_name == 'dqn':
        return QPolicy(buffer_size, gamma, model, SnakeWrapper.action_space, writer, lr=lr)
    else:
        raise Exception('algo {} is not known'.format(policy_name))


def train(steps, buffer_size, opt_every,
          batch_size, lr, max_epsilon, policy_name, gamma, network_name,
          log_dir):

    reset_every = 50
    model = create_model(network_name)
    game = SnakeWrapper()
    writer = SummaryWriter(log_dir=log_dir)
    policy = create_policy(policy_name, buffer_size, gamma, model, writer, lr)

    state = game.reset()
    state_tensor = torch.FloatTensor(state)
    reward_history = []

    for step in tqdm(range(steps)):

        # epsilon exponential decay
        epsilon = max_epsilon * math.exp(-1. * step / (steps / 2))
        writer.add_scalar('training/epsilon', epsilon, step)

        prev_state_tensor = state_tensor
        action = policy.select_action(state_tensor, epsilon)
        state, reward = game.step(action)
        reward_history.append(reward)

        state_tensor = torch.FloatTensor(state)
        reward_tensor = torch.FloatTensor([reward])
        action_tensor = torch.LongTensor([action])

        policy.record(prev_state_tensor, action_tensor, state_tensor, reward_tensor)

        writer.add_scalar('training/reward', reward_history[-1], step)

        if step % opt_every == opt_every - 1:
            policy.optimize(batch_size, step)  # no need for logging, policy logs it's own staff.

        if step % reset_every == reset_every - 1:
            state = game.reset()

    torch.save({'model_state_dict': policy.model.state_dict()}, log_dir + '_' + policy_name + '_model.pkl')
    writer.close()
    # Test game
    test(policy)


def test(policy):
    game = SnakeWrapper()
    state = game.reset()
    for i in range(100):
        action = policy.select_action(torch.FloatTensor(state), 0)
        print(f'the {i} action is {action}')
        state, reward = game.step(action)
        print('rendered board:')
        game.render()
        time.sleep(0.5)


def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--name', type=str, required=True, help='the name of this run')
    p.add_argument('--log_dir', type=str, required=True, help='directory for tensorboard logs (common to many runs)')

    # loop
    p.add_argument('--steps', type=int, default=10000, help='steps to train')
    p.add_argument('--opt_every', type=int, default=100, help='optimize every X steps')

    # opt
    p.add_argument('--buffer_size', type=int, default=800)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('-e', '--max_epsilon', type=float, default=0.3, help='for pg, use max_epsilon=0')
    p.add_argument('-g', '--gamma', type=float, default=.3)
    p.add_argument('-p', '--policy_name', type=str, choices=['dqn', 'pg', 'a2c'], required=True)
    p.add_argument('-n', '--network_name', type=str, choices=['simple', 'dqn', 'small'], required=True)

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


if __name__ == '__main__':
    args = parse_args()
    args.log_dir = os.path.join(args.log_dir, args.name)

    if os.path.exists(args.log_dir):
        if query_yes_no('You already have a run called {}, override?'.format(args.name)):
            shutil.rmtree(args.log_dir)
        else:
            exit(0)

    del args.__dict__['name']
    train(**args.__dict__)


