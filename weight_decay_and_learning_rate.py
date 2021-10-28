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

def adjust_weight_decay_and_learning_rate(network, weight_decay=1e-4, learning_rate=0.1, c=0, skip_list=()):
    """Appropriately applies weight decay to the correct parameters and applies our proposed learning rate reduction on the BN scale parameters by creating different parameter groups for the optimizer.

    Args:
        network: The nn.Module object corresponding to the network to be trained.
        weight_decay: The desired value for weight decay to be applied to convolutional and linear weights.
        learning_rate: The intended initial learning rate for training.
        c: The constant to reduce the initial learning rate by on the BN scale parameters (alpha_{gamma} = alpha / c)
    """
    
    # Create three separate lists for the parameters (that will be treated differently)
    decay = []
    no_decay = []
    no_decay_and_adjusted_lr = []

    # Iterate over all parameters in the network
    for name, param in network.named_parameters():
        # Skip any parameters that do not require gradients
        if not param.requires_grad:
            continue

        # The parameters we care about are 1D in shape (when squeezed), which makes it an easy filter
        if len(param.squeeze().shape) == 1 or name in skip_list:
            # Determines if the parameter is a BN scale/weight or something else
            if 'weight' in name:
                no_decay_and_adjusted_lr.append(param)
                print(f'No weight decay and adjusted learning rate for param: {name}')

            else:
                no_decay.append(param)
                print(f'No weight decay for param: {name}')

        else:
            decay.append(param)

    # Determine whether to adjust the learning rate for BN scale parameters
    if c > 1:
        value = [{ 'params': no_decay, 'weight_decay': 0.0 }, { 'params': no_decay_and_adjusted_lr, 'weight_decay': 0.0, 'lr': learning_rate / c }, { 'params': decay, 'weight_decay': weight_decay }]
    else:
        value = [{ 'params': no_decay, 'weight_decay': 0.0 }, { 'params': no_decay_and_adjusted_lr, 'weight_decay': 0.0 }, { 'params': decay, 'weight_decay': weight_decay }]

    return value