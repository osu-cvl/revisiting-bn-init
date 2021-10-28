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
import os
import argparse
import copy

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

# Other imports
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

# Inner-project imports
from networks import construct_network, NETWORKS
from seeds import make_deterministic, make_complex
from weight_decay_and_learning_rate import adjust_weight_decay_and_learning_rate

def arguments():
    """Obtains the command-line arguments and does some additional processing.

    Returns:
        A dict containing the necessary command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Training arguments')

    # Normal parameters
    parser.add_argument('--path', default='', type=str, metavar='PATH', help='prefix path to images, networks, results')
    parser.add_argument('--name', default='', type=str, metavar='NAME', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='DATA', choices=['cifar10'], help='name of data set')
    parser.add_argument('--network', default='resnet18', type=str, metavar='NET', choices=NETWORKS, help='network architecture')
    parser.add_argument('--batch_size', default=128, type=int, metavar='BS', help='batch size')
    parser.add_argument('--num_epochs', default=180, type=int, metavar='NE', help='number of epochs to train for')
    parser.add_argument('--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--scheduler', default='cos', type=str, metavar='SCH', choices=['none', 'cos'], help='what learning rate scheduler to use')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='MOM', help='optimizer momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='WD', help='optimizer weight decay')

    # Parameters specific to this work
    parser.add_argument('--bn_weight', default=0.1, type=float, metavar='BNW', help='what value to initialize BN scale to to (<= 0 means no scaling)')
    parser.add_argument('--c', default=100, type=float, metavar='C', help='how much to reduce BN scale learning rate by')
    parser.add_argument('--input_norm', default='bn', type=str, metavar='NORM', choices=['bn', 'dataset'], help='what type of input normalization to us')

    # Remaining parameters for reproducibility and specifying the training device
    parser.add_argument('--seed', default=None, type=str, metavar='S', help='rng seed for reproducability')
    parser.add_argument('--device', default='cuda', type=str, metavar='DEV', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], help='device id (e.g. \'cpu\', \'cuda:0\'')
    args = vars(parser.parse_args())

    # If no path is specified, create the default path
    if args['path'] == '':
        args['path'] = './training/'

    # Append the necessary directories
    args['image_dir'] = f'{args["path"]}{args["dataset"]}/images/'
    args['network_dir'] = f'{args["path"]}{args["dataset"]}/networks/'
    args['results_dir'] = f'{args["path"]}{args["dataset"]}/results/'

    # Create those directories if they don't already exist
    if not os.path.isdir(os.path.abspath(args["image_dir"])):
        os.makedirs(os.path.abspath(args["image_dir"]))

    if not os.path.isdir(os.path.abspath(args["network_dir"])):
        os.makedirs(os.path.abspath(args["network_dir"]))

    if not os.path.isdir(os.path.abspath(args["results_dir"])):
        os.makedirs(os.path.abspath(args["results_dir"]))

    # If no seed specified, set to default value of 1 (which will be made more complex later)
    if args['seed'] is None:
        args['seed'] = '1'

    # Create the experiment name (apply default if one is not specified)
    args['name'] = f'{args["dataset"]}_train_weight{args["bn_weight"]}_c{args["c"]}' if args['name'] == '' else args['name']

    # Set the device
    if 'cuda' in args['device']:
        assert torch.cuda.is_available(), 'Device set to GPU but CUDA not available'
    args['device'] = torch.device(args['device'])

    return args

if __name__ == '__main__':
    # Get command line arguments
    args = arguments()

    # Set the rng seed for reproducibility
    make_deterministic(args['seed'])

    # Create the transforms, assuming CIFAR10
    if args['input_norm'] == 'bn':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
    elif args['input_norm'] == 'dataset':
        # Normalization statistics were computed over entire dataset: i.e., batch_size == len(cifar10_dataset)
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4916, 0.4823, 0.4467], std=[0.2472, 0.2437, 0.2618])])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4916, 0.4823, 0.4467], std=[0.2472, 0.2437, 0.2618])])

    # Create the dataset object
    complete_train_dataset = datasets.CIFAR10(root=args['image_dir'], train=True, transform=train_transform, download=True)
    complete_val_dataset = copy.deepcopy(complete_train_dataset)
    complete_val_dataset.transform = test_transform
    test_dataset = datasets.CIFAR10(root=args['image_dir'], train=False, transform=test_transform, download=True)

    # Create the validation set
    # Save the old state so we can have constant val set 
    previous_numpy_state = np.random.get_state()
    previous_torch_state = torch.get_rng_state()
    np.random.seed(make_complex(0))
    torch.manual_seed(make_complex(0))

    # Identify the indexes for train and val instances
    train_indexes, val_indexes = train_test_split(np.arange(len(complete_train_dataset)), test_size=0.1, stratify=complete_train_dataset.targets)

    # Restore the previous random state
    np.random.set_state(previous_numpy_state)
    torch.set_rng_state(previous_torch_state)

    # Create the dataset subsets
    train_dataset = data.Subset(complete_train_dataset, train_indexes)
    val_dataset = data.Subset(complete_val_dataset, val_indexes)

    # Free up some sweet, sweet memory
    del complete_train_dataset, complete_val_dataset

    # Create the data loaders
    train_dataloader = data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True)

    # Set up the rest of training
    num_classes = len(train_dataset.dataset.classes)
    network = construct_network(network_name=args['network'], num_classes=num_classes, dataset=args['dataset'], bn_weight=args['bn_weight'], input_norm=args['input_norm'])
    network = network.to(args['device'])

    # Instantiate the loss function
    loss_func = nn.CrossEntropyLoss()

    # If a value for weight decay or a value for reducing the BN scale learning rate has been specified, correctly apply it
    if args['weight_decay'] > 0 or args['c'] > 1:
        parameters = adjust_weight_decay_and_learning_rate(network, weight_decay=args['weight_decay'], learning_rate=args['learning_rate'], c=args['c'])
    else:
        parameters = network.parameters()

    # Instantiate the optimizer
    optimizer = optim.SGD(parameters, lr=args['learning_rate'], momentum=args['momentum'])

    # Instantiate the learning rate scheduler (if specified)
    if args['scheduler'] == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'], eta_min=1e-8)
    elif args['scheduler'] == 'none':
        scheduler = None

    # Identify the number of training, validation, and test examples
    num_train_instances = len(train_dataset)
    num_val_instances = len(val_dataset)
    num_test_instances = len(test_dataset)

    # Identify the number of training, validation, and test batches
    num_train_batches = len(train_dataloader)
    num_val_batches = len(val_dataloader)
    num_test_batches = len(test_dataloader)

    # Variable for keeping track of the best network weights
    best_validation_accuracy = 0.0

    # Arrays for keeping track of epoch results
    # Train
    training_losses = np.zeros(args['num_epochs'])

    # validation
    validation_losses = np.zeros(args['num_epochs'])
    validation_accuracies = np.zeros(args['num_epochs'])

    # Test
    test_accuracies = np.zeros(args['num_epochs'])

    # Training
    for epoch in range(args['num_epochs']):
        
        # Print out the epoch number
        print(f'Epoch {epoch}:')
        
        # Prepare for training by enabling gradients
        network.train()
        torch.set_grad_enabled(True)
        
        # Instantiate the running training loss
        training_loss = 0.0
        
        # Iterate over the TRAINING batches
        for batch_num, (images, labels) in enumerate(train_dataloader):
            
            # Send images and labels to compute device
            images = images.to(args['device'])
            labels = labels.to(args['device'])
            
            # Zero the previous gradients
            optimizer.zero_grad()
                    
            # Forward propagation
            predictions = network(images)
            
            # Compute loss
            loss = loss_func(predictions, labels)
            
            # Backward propagation
            loss.backward()
            
            # Adjust weights
            optimizer.step()
            
            # Accumulate average loss
            training_loss += loss.item()
            
            # Give epoch status update
            print(' ' * 100, end='\r', flush=True) 
            print(f'Epoch {epoch}: {100. * (batch_num + 1) / num_train_batches : 0.1f}% ({batch_num + 1}/{num_train_batches}) - Loss = {loss.item()}', end='\r', flush=True)
        
        # Clear the status update message
        print(' ' * 100, end='\r', flush=True) 
        
        # Get the average training loss
        training_loss /= num_train_batches
        print(f'Training Loss: {training_loss : 0.6f}')

        # Take a LR scheduler step
        if scheduler is not None:
            scheduler.step()

        # Disable computing gradients
        network.eval()
        torch.set_grad_enabled(False)
        
        # Instantiate two arrays for keeping track of ground truth labels and predicted labels
        true_classes = np.empty(num_val_instances)
        predicted_classes = np.empty(num_val_instances)
        validation_loss = 0.0
        
        # Iterate over the VALIDATION batches
        for batch_num, (images, labels) in enumerate(val_dataloader):
            
            # Send images and labels to compute device
            images = images.to(args['device'])
            labels = labels.to(args['device'])
            
            # Forward propagation
            predictions = network(images)

            # Get the validation loss
            loss = F.cross_entropy(predictions, labels, reduction='sum')
            validation_loss += loss.item()
            
            # Threshold for flat prediction
            _, predictions = torch.max(predictions, 1)

            # Get the correct flags
            correct = torch.squeeze(predictions == labels)
            
            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * args['batch_size'] : min( (batch_num + 1) * args['batch_size'], num_val_instances) ] = labels.detach().cpu().numpy().astype(int)
            predicted_classes[ batch_num * args['batch_size'] : min( (batch_num + 1) * args['batch_size'], num_val_instances) ] = predictions.detach().cpu().numpy().astype(int)

            # Give epoch status update
            print(' ' * 100, end='\r', flush=True) 
            print(f'Validation: {100. * (batch_num + 1) / num_val_batches : 0.1f}% ({batch_num + 1}/{num_val_batches})', end='\r', flush=True)
        
        # Clear the status update message
        print(' ' * 100, end='\r', flush=True) 

        # Get the average training loss
        validation_loss /= num_val_instances
        print(f'Validation Loss: {validation_loss : 0.6f}')
        
        # Compute validation set accuracy
        validation_accuracy = metrics.accuracy_score(true_classes, predicted_classes)
        print(f'Validation Accuracy: {validation_accuracy * 100.0 : 0.3f}')
        
        # Save the new best network weights if validation accuracy improved
        if validation_accuracy > best_validation_accuracy:
            print('Found improved network')
            
            best_validation_accuracy = validation_accuracy

            torch.save(network.state_dict(), f'{args["network_dir"]}/{args["name"]}-best.pt')
        
        # Disable computing gradients
        network.eval()
        torch.set_grad_enabled(False)
        
        # Instantiate two arrays for keeping track of ground truth labels and predicted labels
        true_classes = np.empty(num_test_instances)
        predicted_classes = np.empty(num_test_instances)
        
        # Iterate over the TEST batches
        for batch_num, (images, labels) in enumerate(test_dataloader):
            
            # Send images and labels to compute device
            images = images.to(args['device'])
            labels = labels.to(args['device'])
            
            # Forward propagation
            predictions = network(images)
            
            # Threshold for flat prediction
            _, predictions = torch.max(predictions, 1)

            # Get the correct flags
            correct = torch.squeeze(predictions == labels)
            
            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * args['batch_size'] : min( (batch_num + 1) * args['batch_size'], num_test_instances) ] = labels.detach().cpu().numpy().astype(int)
            predicted_classes[ batch_num * args['batch_size'] : min( (batch_num + 1) * args['batch_size'], num_test_instances) ] = predictions.detach().cpu().numpy().astype(int)

            # Give epoch status update
            print(' ' * 100, end='\r', flush=True) 
            print(f'Testing: {100. * (batch_num + 1) / num_test_batches : 0.1f}% ({batch_num + 1}/{num_test_batches})', end='\r', flush=True)
        
        # Clear the status update message
        print(' ' * 100, end='\r', flush=True) 
        
        # Compute test set accuracy
        test_accuracy = metrics.accuracy_score(true_classes, predicted_classes)
        print(f'Test Accuracy: {test_accuracy * 100.0 : 0.3f}')

        # Save epoch results
        training_losses[epoch] = training_loss
        validation_losses[epoch] = validation_loss
        validation_accuracies[epoch] = validation_accuracy * 100.0
        test_accuracies[epoch] = test_accuracy * 100.0

    try:
        # Output training to file
        with open(f'{args["results_dir"]}/{args["name"]}.txt', 'w') as f:
            # Write out general parameters
            f.write(f'Name: {args["name"]},,,, \n')
            f.write(f'Dataset: {args["dataset"]},,,, \n')
            f.write(f'Network: {args["network"]},,,, \n')
            f.write(f'Batch Size: {args["batch_size"]},,,, \n')
            f.write(f'Num Epochs: {args["num_epochs"]},,,, \n')
            f.write(f'Learning Rate: {args["learning_rate"]},,,, \n')
            f.write(f'LR Scheduler: {args["scheduler"]},,,, \n')
            f.write(f'Momentum: {args["momentum"]},,,, \n')
            f.write(f'Weight Decay: {args["weight_decay"]},,,, \n')

            # Write out specifics for this work
            f.write(f'BN Intial Scale: {args["bn_weight"]},,,, \n')
            f.write(f'LR Reduction Value: {args["c"]},,,, \n')
            f.write(f'Input Normalization Method: {args["input_norm"]},,,, \n')

            # Write out header line specifying each column
            f.write('epoch,train_loss,val_loss,val_accuracy,test_accuracy \n')

            zip_object = zip(training_losses, validation_losses, validation_accuracies, test_accuracies)

            # Write out the values for each epoch
            for epoch, (train_loss, val_loss, val_acc, test_acc) in enumerate(zip_object):
                f.write(f'{epoch},{train_loss},{val_loss},{val_acc},{test_acc} \n')
    except:
        print('error when saving training results')



        