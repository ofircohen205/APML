import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch
import random
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from base_policy import BasePolicy
from train_snake import *


class ActorCriticModel(nn.Module):

    def __init__(self, h=9, w=9, inputs=10, outputs=3):
        """
        :param h:
        :param w:
        :param outputs: number of actions (3)
        """
        super(ActorCriticModel, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return F.softmax(self.head(x.view(x.size(0), -1)), dim=-1)

class ActorCriticPolicy(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(ActorCriticPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.episodes = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.reward_history = []
        self.total_rewards = []
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        pass

    def optimize(self, batch_size, global_step=None, alpha=None):
        pass

    def discount_rewards(self, rewards, gamma=0.99):
        pass


def train_policy(steps, buffer_size, opt_every, batch_size, lr, max_epsilon, policy_name, gamma, network_name, log_dir):
    model = create_model(network_name)
    game = SnakeWrapper()
    writer = SummaryWriter(log_dir=log_dir)
    state = game.reset()
    policy = ActorCriticPolicy(buffer_size, gamma, model, game.action_space, writer, lr)
    alpha = 0.3
    total_deaths = 0
    reset_game = False

    for step in tqdm(range(steps)):
        if reset_game:
            print("Game has been reset")
            game.reset()
            reset_game = False

        epsilon = max_epsilon * math.exp(-1. * step / (steps / 2))
        writer.add_scalar('training/epsilon', epsilon, step)
        action = policy.select_action(torch.FloatTensor(state), epsilon)
        state, reward = game.step(action)
        while reward != -5 and policy.episodes <= 200:
            policy.states.append(state[0])
            policy.rewards.append(reward)
            policy.actions.append(action)
            state, reward = game.step(action)
            policy.episodes += 1
            if reward == -5:
                total_deaths += 1
                reset_game = True
                print("Episodes done. total steps: {}".format(policy.episodes))

        policy.optimize(batch_size, step, alpha=alpha)  # no need for logging, policy logs it's own staff.
        policy.episodes = 0

    print("Total deaths in training: {}".format(total_deaths))
    torch.save({'model_state_dict': policy.model.state_dict()}, log_dir + '_' + policy_name + '_model.pkl')
    writer.close()
    # Test game
    test(policy)


if __name__ == '__main__':
    args = parse_args()
    args.log_dir = os.path.join(args.log_dir, args.name)

    if os.path.exists(args.log_dir):
        if query_yes_no('You already have a run called , override?'.format(args.name)):
            shutil.rmtree(args.log_dir)
        else:
            exit(0)

    del args.__dict__['name']
    train_policy(**args.__dict__)

