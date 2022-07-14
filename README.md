# Revisiting Batch Norm Initialization

This repo contains the official code for the paper ["Revisiting Batch Norm Initialization"](https://arxiv.org/abs/2110.13989) by [Jim Davis](http://web.cse.ohio-state.edu/~davis.1719/) and [Logan Frank](https://loganfrank.github.io/), which was accepted to the European Conference on Computer Vision (ECCV) 2022.

In this work, we observed that the learned scale (&gamma;) and shift (β) affine transformation parameters of batch normalization tend to remain close to their initialization and further noticed that the normalization operation of batch normalization can yield overly large values, which are preserved through the remainder of the forward pass due to the previous observation. We first examined the batch normalization gradient equations and derived the influence of the batch normalization scale parameter with respect to training then empirically showed across multiple datasets and network architectures that with initializations of the BN scale parameter < 1 and reducing the learning rate on the batch normalization scale parameter, statistically significant gains in performance can be seen (according to a one-sided paired t-test).

## Overview 

The contents of this repo are organized as follows:
* [train.py](train.py): a sample script for training a ResNet model on CIFAR-10 that employs our proposed initialization and learning rate reduction methods.
* [batch_norm.py](batch_norm.py): a class that extends the existing PyTorch implementation for BN and implements our proposed initialization method.
* [weight_decay_and_learning_rate.py](weight_decay_and_learning_rate.py): a function that separates the network parameters into different parameter groups; all biases receive no weight decay and the specified learning rate, BN scale parameters receive no weight decay and a (optionally) reduced learning rate, and all other parameters (conv and linear weights) receive the specified weight decay and learning rate. This function contains the code for our learning rate reduction method.
* [seeds.py](seeds.py): two functions that we employ to 1) create more complex seeds (from simple integers) and 2) seed various packages / RNGs for reproducibility.
* [t_test.py](t_test.py): the function we use to compute our one-sided paired t-test for evaluating statistical significance of results.
* [networks.py](networks.py): two functions for initializing a network and setting all biases to 0; another function for instantiating the network, making necessary modifications (e.g., for CIFAR-10), calling the initialization functions, and prepending our proposed input normalization BN layer (if desired).

## Requirements

Assuming you have already created an environment with Python 3.8 and pip, install the necessary package requirements in [requirements.txt](requirements.txt) using ```pip install -r requirements.txt```. The main requirements are

* PyTorch
* TorchVision
* PIL
* NumPy
* SciKit-Learn
* SciPy

with specific versions given in [requirements.txt](requirements.txt).

## Training

An example for running our training algorithm using everything proposed in the paper is:

```
python train.py \
    --path '' \
    --name 'train_example' \
    --dataset 'cifar10' \
    --network 'resnet18' \
    --batch_size 128 \
    --num_epochs 180 \
    --learning_rate 0.1 \
    --scheduler 'cos' \
    --momentum 0.9 \
    --weight_decay 0.0001 \
    --device 'cuda:0' \
    --bn_weight 0.1 \
    --c 100 \
    --input_norm 'bn' \
    --seed '1' 
```

where 

* ```path``` is the location to the parent directory where your data is (or will be) stored, ```${path}/images/```, where you would like to save the network weights, ```${path}/networks/```, and where you would like to save the output file, ```${path}/results/```. If an empty string is provided, the directories will be created in this directory.
* ```name``` is the name you would like to give for the network weights file and output file (e.g., ```train_example-best.pt``` is the resulting network weights).
* ```dataset``` is the name of the desired dataset to be trained on. CIFAR10 is the only dataset supported in this training script as it can be downloaded using torchvision if not already present in ```${path}/images/```. Other datasets provided by torchvision can be easily implemented as well. If providing your own dataset (e.g., CUB200), you must place the data in ```${path}/images/``` and implement it in the code (as a ```torchvision.datasets.ImageFolder``` or a custom class).
* ```network``` is the name of the network that will be trained in this script. All base ResNet architectures are supported in this training script using torchvision. Other network architectures implemented in torchvision can be easily implemented for use (minus any modifications that need to be made for the small imagery in CIFAR10). Custom network architectures can also be easily imported and implemented for use in this training script.
* ```batch_size``` is the number of examples in a training batch and for the sake of speed, is the number of examples in a batch of validation or test data.
* ```num_epochs``` is the total number of training epochs.
* ```learning_rate``` is the initial learning rate for training, which can change according to a learning rate scheduler.
* ```scheduler``` is the desired learning rate scheduler, ```'none'``` (i.e., a constant learning rate) and ```'cos'``` are the only supported options in this training script.
* ```momentum``` is the momentum coefficient used in the stochastic gradient descent optimizer.
* ```weight_decay``` is the weight decay coefficient used in the stochastic gradient descent optimizer.
* ```device``` specifies the desired compute device. Examples include ```'cpu'```, ```'cuda'```, and ```'cuda:0'```.

The above command-line arguments are general arguments for training a CNN. The next four command-line arguments are specific to our work where

* ```bn_weight``` is the constant value that the scale parameter in all batch normalization layers in the network will be initialized to. To employ our batch normalization scale parameter initialization, set this value to <1 (0.1 seems to be a good starting point for finding the optimal value).
* ```c``` is the constant value used to reduce the learning rate on all batch normalization scale parameters. For all batch normalization scale parameters in the network, the learning rate applied to these parameters will be divided by ```c``` and will still follow the learning rate scheduler. In our work, we used a value of 100, but significant gains can be seen as long as this value is sufficiently and reasonably large.
* ```input_norm``` specifies how the input data will be normalized. Using a value of ```'bn'``` employs our proposed batch normalization input normalization technique. The only other supported option is ```'dataset'``` which uses the statistics of CIFAR10 computed globally.
* ```seed``` is the value that will be used to set all random number generator seeds. Whatever this value is, it will be made more complex using MD5 (complex being a sufficiently large value with a balanced mix of 0's and 1's in binary representation) and then used for seeding the random number generators. Values of [0, 88265] will yield unique integer seeds. We used [1, 15] for our work with 0 to create the validation set.

## Batch Normalization

To instantiate a single batch normalization layer using the ```ScaleBatchNorm2d``` class in [batch_norm.py](batch_norm.py), call

```
bn1 = ScaleBatchNorm2d(num_feature=64, eps=1e-5, momentum=0.1, affine=True, bn_weight=0.1)
```

which creates a batch normalization layer that takes 64 feature maps as input and initializes the scale parameter to a value of 0.1.

In many cases, a partial function may be useful (e.g., when calling the torchvision constructors for ResNet, etc.). An example of creating a partial function then using that partial function is

```
norm_layer = partial(ScaleBatchNorm2d, eps=1e-5, momentum=0.1, affine=True, bn_weight=0.1)
network = torchvision.models.resnet18(num_classes=num_classes, norm_layer=norm_layer)
```

which creates a ResNet18 network from the torchvision library that utilizes our proposed ```ScaleBatchNorm2d```.

## Network

To create a network using our ```construct_network``` function in [networks.py](networks.py), call

```
network = construct_network(network_name='resnet18', num_classes=10, dataset='cifar10', bn_weight=0.1, input_norm='bn')
```

The above function call will instantiate a ResNet18 network with 10 output classes, is modified to account for the smaller imagery of CIFAR10, initializes all batch normalization layers to have an initial scale value of 0.1, and utilizes our proposed batch normalization-based input normalization scheme. ```network_name``` can be any of the base ResNet architectures (18, 34, 50, 101, 152) following a similar string value as provided, ```bn_weight``` can be any value >0 (though our work proposes setting this value <1), and ```input_norm``` can be ```bn``` to employ our proposed input normalization or ```dataset``` to employ the precomputed global dataset statistics for CIFAR10.

## Reducing Learning Rate

Given a network has been instantiated, the learning rate for the batch normalization scale parameters can be reduced and provided to an optimizer using the following example. Note this example will also properly apply weight decay to only the convolutional and fully-connected weights. All network biases and batch normalization parameters will have weight decay == 0.

```
parameters = adjust_weight_decay_and_learning_rate(network, weight_decay=1e-4, learning_rate=0.1, c=100)
optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9)
```

## Setting Seeds

Setting the seeds for important random number generators using our functions provided in [seeds.py](seeds.py) is easily done by calling

```
make_deterministic('1')
```

which will take the value of 1, make it more complex using MD5, then seed various random number generators using that complex value.

## Evaluating Significance

Determining whether improvements are significant is crucial, this can be done by calling the ```evaluate``` function in [t_test.py](t_test.py). For example,

```
my_cool_new_method = np.array([91.7, 93.4, 92.2, 90.0, 91.9])
baseline = np.array([91.0, 92.7, 92.2, 90.1, 91.5])

p_value = evaluate(my_cool_new_method, baseline)

if p_value <= 0.05:
    print('My approach is significantly greater than the baseline!')
```

## Citation

Please cite our paper "Revisiting Batch Norm Initialization" with

```
@article{Davis2022revisiting,
  title={Revisiting Batch Norm Initialization},
  author={Davis, Jim and Frank, Logan},
  journal={European Conference on Computer Vision},
  year={2022}
}
```
