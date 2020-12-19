import torch
import torch.nn.functional as F
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym


class QPolicy(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(QPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)

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
            pass
        else:
            return self.action_space.sample()  # return action randomly

    def optimize(self, batch_size, global_step=None):
        # optimize your model

        if len(self.memory) < batch_size:
            return None

        self.memory.batch_size = batch_size
        for transitions_batch in self.memory:

            # transform list of tuples into a tuple of lists.
            # explanation here: https://stackoverflow.com/a/19343/3343043
            batch = Transition(*zip(*transitions_batch))

            state_batch = torch.cat(batch.state)
            next_state_batch = torch.cat(batch.next_state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # do your optimization magic here!
            # YOUR CODE HERE!

