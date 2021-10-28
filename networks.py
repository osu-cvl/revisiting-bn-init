"""
Paper: Revisiting Batch Normalization
arXiv Link: https://arxiv.org/pdf/2110.13989.pdf
Authors: Jim Davis and Logan Frank*
Affiliation: Department of Computer Science & Engineering, Ohio State University
Corresponding Email: frank.580@osu.edu (First: Logan, Last: Frank)
Date: Oct 25, 2021

This research was supported by the U.S. Air Force Research Laboratory under Contract #GRT00054740 (Release #AFRL-2021-3711). 
We would also like to thank the DoD HPCMP for the use of their computational resources.
"""

# Standard Python imports
import math
from functools import partial

# PyTorch imports
import torch.nn as nn
import torchvision.models as models

# Inner-project imports
from batch_norm import ScaleBatchNorm2d

# List showing all supported network architectures
NETWORKS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def reinit_network(network, fan='in', activation_function=nn.ReLU):
    """Reinitializes the weights in a network.

    Args:
        network: The nn.Module object containing the network weights.
        fan: Whether to initialize according to the fan in or fan out of each layer.
        activation_function: Which activation function is used in the network (choices are ReLU or LeakyReLU since others are not supported).
    """

    mode = 'fan_in' if fan == 'in' else 'fan_out'
    nonlinearity = 'relu' if activation_function == nn.ReLU else 'leaky_relu'

    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity, a=(0 if nonlinearity == 'relu' else 0.01))
        elif isinstance(m, nn.Linear):
            # This is the default initialization of a nn.Linear layer in PyTorch
            nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity, a=math.sqrt(5))

def zero_biases(network):
    """Sets all biases (both nn.Linear and BN) to 0.

    Args:
        network: The nn.Module object containing the network weights.
    """

    for m in network.modules():
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def construct_network(network_name='resnet18', num_classes=10, dataset='', bn_weight=1.0, input_norm='bn'):
    """Creates a network from the torchvision package then does additional adjustments to the network.

    Creates a nn.Module object according to a network architecture definition in PyTorch, performs modifications (if necessary, e.g., for CIFAR10 sized images),
    reinitializes the weights and sets biases to 0, and appends a input normalization BN layer (if desired).

    Args:
        network_name: The string name of the desired network architecture.
        num_classes: The number of classes in the training dataset.
        dataset: The string name of the dataset that will be used for training.
        bn_weight: Float value to initialize the scale in all BN layers to. A value 0 < gamma < 1 employs our method.
        input_norm: Specifies how the input data will be normalized before the first convolutional layer.

    Returns:
        A nn.Module object that corresponds to a network architecture derived from the ones in the torchvision package.
    """

    # Create a partial function for our BN layer with scaling
    norm_layer = partial(ScaleBatchNorm2d, eps=1e-5, momentum=0.1, bn_weight=bn_weight)

    # Create the network from torchvision
    if network_name == 'resnet18':
        network = models.resnet18(num_classes=num_classes, norm_layer=norm_layer)
    elif network_name == 'resnet34':
        network = models.resnet34(num_classes=num_classes, norm_layer=norm_layer)
    elif network_name == 'resnet50':
        network = models.resnet50(num_classes=num_classes, norm_layer=norm_layer) 
    elif network_name == 'resnet101':
        network = models.resnet101(num_classes=num_classes, norm_layer=norm_layer)
    elif network_name == 'resnet152':
        network = models.resnet152(num_classes=num_classes, norm_layer=norm_layer)   
        
    # Modify network for CIFAR10-sized images if needed, other change for Tiny ImageNet sized images
    if dataset in ('cifar10') and 'resnet' in network_name:
        network.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        network.maxpool = nn.Identity()

    # Reinitialize the network layers
    reinit_network(network, fan='in')

    # Set biases to 0
    zero_biases(network)

    # Add before network batch norm layer if needed
    if input_norm == 'bn':
        # Instantiate the value for sigma_{act}, see paper for details 
        sigma_act = 0.58

        # Create the input data normalization BN layer
        input_norm_bn = ScaleBatchNorm2d(3, eps=1e-5, momentum=0.1, bn_weight=(bn_weight * sigma_act), affine=True)
        input_norm_bn.bias.requires_grad = False

        # Prepend the input data normalization BN layer to the original network
        network = nn.Sequential(input_norm_bn, network)

    return network    