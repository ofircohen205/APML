import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.




