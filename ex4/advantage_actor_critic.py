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
from tqdm import trange

from base_policy import BasePolicy
from snake import THE_DEATH_PENALTY
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

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.critic_head = nn.Linear(linear_input_size, outputs)
        self.actor_head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        value = self.critic_head(x)
        policy_dist = F.softmax(self.actor_head(x), dim=-1)
        return value, policy_dist


class ActorCriticPolicy(BasePolicy):
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(ActorCriticPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.max_episode_num = 500
        self.num_actions = 3
        self.nums_of_steps = []
        self.avg_num_of_steps = []
        self.all_rewards = []
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropy_term = torch.FloatTensor([0])

    def select_action(self, state, epsilon, global_step=None, alpha=None):
        value, policy_dist = self.model(state)
        return value, policy_dist

    def optimize(self, global_step, alpha=None):
        # compute Q values
        Qval = self.values.pop()
        Qvals = torch.zeros(len(self.values))
        for t in reversed(range(len(self.rewards))):
            Qval = self.rewards[t] + self.gamma * Qval
            Qvals[t] = Qval[0, 0]

        # update actor critic
        values = torch.cat(self.values)
        log_probs = torch.stack(self.log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        # critic_loss = F.mse_loss(target=Qvals, input=values)
        loss = actor_loss + critic_loss + alpha * self.entropy_term

        self.optimizer.zero_grad()
        loss.backward()
        self.writer.add_scalar('training/loss', loss.item(), global_step)
        self.optimizer.step()
        self.clear_buffer()

    def clear_buffer(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropy_term = torch.FloatTensor([0])


def train_policy(steps, buffer_size, opt_every, batch_size, lr, max_epsilon, policy_name, gamma, network_name, log_dir):
    a2c_model = ActorCriticModel()
    game = SnakeWrapper()
    writer = SummaryWriter(log_dir=log_dir)
    policy = ActorCriticPolicy(buffer_size, gamma, a2c_model, game.action_space, writer, lr)
    all_lengths = []
    average_lengths = []
    all_rewards = []

    prog_bar = trange(policy.max_episode_num, desc='', leave=True)
    for episode in prog_bar:
        state = game.reset()

        epsilon = max_epsilon * math.exp(-1. * episode / (steps / 2))
        writer.add_scalar('training/epsilon', epsilon, episode)

        for step in range(steps):
            value, policy_dist = policy.select_action(torch.FloatTensor(state), epsilon)
            dist = Categorical(probs=policy_dist)

            action = dist.sample()
            log_prob = dist.log_prob(value=action)
            entropy = dist.entropy()
            new_state, reward = game.step(action.item())

            policy.log_probs.append(log_prob)
            policy.rewards.append(reward)
            policy.values.append(value[0, action])
            policy.entropy_term += entropy
            state = new_state

            if reward == THE_DEATH_PENALTY:
                q_value, _ = policy.model(torch.FloatTensor(state))
                policy.values.append(q_value)
                all_rewards.append(np.sum(policy.rewards))
                all_lengths.append(step)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 1 == 0:
                    writer.add_scalar('training/episode_length', step, episode)
                    writer.add_scalar('training/episode_rewards', sum(policy.rewards), episode)
                    prog_bar.set_description(
                        "episode: {}, total reward: {}, average_reward: {}, length: {}".format
                        (episode, np.round(np.sum(policy.rewards), decimals=3), np.round(
                            np.mean(policy.all_rewards[-10:]), decimals=3), step))
                break

            state = new_state
            policy.nums_of_steps.append(step)

        policy.avg_num_of_steps.append(np.mean(policy.nums_of_steps[-10:]))
        policy.all_rewards.append(np.sum(policy.rewards))
        policy.optimize(global_step=episode, alpha=0.001)

    writer.close()
    # policy_test(policy)


def policy_test(policy):
    game = SnakeWrapper()
    state = game.reset()
    total_rewards = []
    total_deaths = 0
    for i in range(100):
        value, policy_dist = policy.select_action(torch.FloatTensor(state), 0)
        dist = Categorical(probs=policy_dist)
        action = dist.sample().item()
        print(f'the {i} action is {action}')
        state, reward = game.step(action)
        if reward == -5:
            total_deaths += 1
        total_rewards.append(reward)
        print('rendered board:')
        game.render()
        print("rewards so far for player 1: {}".format(sum(total_rewards)))
        time.sleep(0.5)
    print("Total deaths: {}".format(total_deaths))


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

