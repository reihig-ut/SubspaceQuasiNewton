import jax.nn as nn

from utils.jax_layers import cross_entropy_loss


def get_activation(activation_name):
    if activation_name == "sigmoid":
        activation = nn.sigmoid
    elif activation_name == "relu":
        activation = nn.relu
    elif activation_name == "elu":
        activation = nn.elu
    elif activation_name == "gelu":
        activation = nn.gelu
    elif activation_name == "tanh":
        activation = nn.tanh
    elif activation_name == "celu":
        activation = nn.celu
    else:
        raise ValueError("No activation")
    return activation


def get_criterion(criterion_name):
    if criterion_name == "CrossEntropy":
        criterion = cross_entropy_loss
    else:
        raise ValueError("No criterion")
    return criterion
