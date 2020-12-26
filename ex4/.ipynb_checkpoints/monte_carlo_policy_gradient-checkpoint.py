import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch
import random
import gym
import numpy as np
from torch.distributions import Bernoulli, Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from base_policy import BasePolicy
from snake import THE_DEATH_PENALTY
from train_snake import *
from itertools import count


class MonteCarloPolicy(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(MonteCarloPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.max_episode_num = 1000
        self.nums_of_steps = []
        self.avg_num_of_steps = []
        self.all_rewards = []
        self.probs = []
        self.rewards = []
        self.actions = []
        self.optimizer = optim.Adam(self.model.parameters())

    def select_action(self, state, epsilon, global_step=None):
        action_probs = self.model(state)
        return Categorical(probs=action_probs).sample().item(), action_probs

    def optimize(self, global_step, alpha=None, normalize=False):
        objective = self.calculate_objective(alpha=alpha, normalize=normalize)
        loss = -objective.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.writer.add_scalar('training/loss', loss.item(), global_step)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
        self.optimizer.step()

    def calculate_objective(self, alpha, normalize):
        discounted_rewards = self.calculate_rewards(normalize)
        objective = []
        for t in range(len(self.rewards)):
            objective.append(torch.log(self.probs[t][0][self.actions[t]]) * discounted_rewards[t] + alpha * Categorical(
                probs=self.probs[t]).entropy())

        objective = torch.cat(objective)
        return objective

    def calculate_rewards(self, normalize):
        discounted_rewards = []
        for t in range(len(self.rewards)):
            Gt = 0
            pw = 0
            for r in self.rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        if normalize:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        return discounted_rewards

    def clear_buffer(self):
        self.probs = []
        self.rewards = []
        self.actions = []


def train_monte_carlo_policy(steps, buffer_size, opt_every, batch_size, lr, max_epsilon, policy_name, gamma,
                             network_name, log_dir):
    model = create_model(network_name)
    game = SnakeWrapper()
    writer = SummaryWriter(log_dir=log_dir)
    policy = MonteCarloPolicy(buffer_size, gamma, model, game.action_space, writer, lr)
    alpha = .1

    prog_bar = trange(policy.max_episode_num, desc='', leave=True)
    for episode in prog_bar:
        state = game.reset()
        state = torch.FloatTensor(state)
        policy.clear_buffer()

        for step in range(steps):
            action, prob = policy.select_action(state, 0)
            state, reward = game.step(action)
            state = torch.FloatTensor(state)
            policy.probs.append(prob)
            policy.rewards.append(reward)
            policy.actions.append(action)

            if reward == THE_DEATH_PENALTY:
                writer.add_scalar('training/episode_length', step, episode)
                writer.add_scalar('training/episode_rewards', sum(policy.rewards), episode)
                prog_bar.set_description("Total steps: {}, total reward: {}, average_reward: {}".format
                                         (step, np.round(np.sum(policy.rewards), decimals=3), np.round(
                                             np.mean(policy.all_rewards[-5:]), decimals=3)))
                break
            policy.nums_of_steps.append(step)

        policy.avg_num_of_steps.append(np.mean(policy.nums_of_steps[-5:]))
        policy.all_rewards.append(np.sum(policy.rewards))
        policy.optimize(global_step=episode, alpha=alpha)

    writer.close()
    test(policy)


if __name__ == '__main__':
    args = parse_args()
    args.log_dir = os.path.join(args.log_dir, args.name)

    if os.path.exists(args.log_dir):
        if query_yes_no('You already have a run called {}, override?'.format(args.name)):
            shutil.rmtree(args.log_dir)
        else:
            exit(0)

    del args.__dict__['name']
    train_monte_carlo_policy(**args.__dict__)
