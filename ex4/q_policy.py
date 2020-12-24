import torch
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
from models import SimpleModel, DqnModel
import gym


class QPolicy(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(QPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.target_model = DqnModel()
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.i_episode = 0

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
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)[0].item()
        else:
            return self.action_space.sample()  # return action randomly

    def optimize(self, batch_size, global_step=None):
        # optimize your model
        if len(self.memory) < batch_size:
            return None

        self.target_model.eval()
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
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.model(state_batch).gather(1, action_batch.reshape(batch_size, 1))

            # Compute the expected Q values
            with torch.no_grad():
                next_state_values = self.model(next_state_batch).max(1)[0].detach()
                expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.writer.add_scalar('training/loss', loss.item(), global_step)
            # for param in self.model.parameters():
            #     param.grad.data.clamp_(-1, 1)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.optimizer.step()

