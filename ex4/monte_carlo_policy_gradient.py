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
from scipy.stats import entropy
from base_policy import BasePolicy
from train_snake import *


class MonteCarloPolicy(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(MonteCarloPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def select_action(self, state, epsilon, global_step=None):
        """
                select an action to play
                :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
                :param epsilon: epsilon...
                :param global_step: used for tensorboard logging
                :return: return single action as integer (0, 1 or 2).
                """

        random_number = random.random()
        if random_number > epsilon:
            # here you do the action-selection magic!
            # YOUR CODE HERE
            return self.model(state).max(1)[1].view(1, 1)[0].item()
        else:
            return self.action_space.sample()  # return action randomly


    def optimize(self, batch_size, global_step=None, alpha=None):
        print("Start optimizing")
        self.batch_rewards.extend(self.discount_rewards(self.rewards))
        self.batch_states.extend(self.states)
        self.batch_actions.extend(self.actions)
        self.batch_counter += 1
        self.total_rewards.append(sum(self.rewards))

        self.optimizer.zero_grad()
        state_tensor = torch.FloatTensor(self.batch_states)
        reward_tensor = torch.FloatTensor(self.batch_rewards)
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(self.batch_actions)

        # Calculate loss
        prob = self.model(state_tensor)
        logprob = torch.log(prob)
        entropy_factor = alpha * (-torch.sum(prob * logprob))
        selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor] + entropy_factor
        loss = -selected_logprobs.mean()

        # Calculate gradients
        loss.backward()
        self.writer.add_scalar('training/loss', loss.item(), global_step)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
        # Apply gradients
        self.optimizer.step()

        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1
        print("End optimizing")

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return (r - np.mean(r)) / np.std(r)


def train_policy(steps, buffer_size, opt_every, batch_size, lr, max_epsilon, policy_name, gamma, network_name, log_dir):
    reset_every = 500
    model = create_model(network_name)
    game = SnakeWrapper()
    writer = SummaryWriter(log_dir=log_dir)
    state = game.reset()
    policy = MonteCarloPolicy(buffer_size, gamma, model, game.action_space, writer, lr)

    for step in tqdm(range(steps)):
        epsilon = max_epsilon * math.exp(-1. * step / (steps / 2))
        writer.add_scalar('training/epsilon', epsilon, step)
        action = policy.select_action(torch.FloatTensor(state), epsilon)
        state, reward = game.step(action)
        while reward != -5 and policy.episodes <= 2000:
            policy.states.append(state[0])
            policy.rewards.append(reward)
            policy.actions.append(action)
            state, reward = game.step(action)
            policy.episodes += 1
            if reward == -5:
                print("Died after {} frames".format(policy.episodes))

        policy.optimize(batch_size, step, epsilon)  # no need for logging, policy logs it's own staff.
        policy.episodes = 0

        # if step % reset_every == reset_every - 1:
        #     state = game.reset()

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

