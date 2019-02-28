
# Revisiting Data Augmentation for Rotational Invariance in Convolutional Neural Networks

This repository contains the code necessary to obtain the experimental results published in the article *Revisiting Data Augmentation for Rotational Invariance in Convolutional Neural Networks* (link and bib entry coming soon).

## Abstract
`Convolutional Neural Networks (CNN) offer state of the art performance in various computer vision tasks. Many of those tasks require different subtypes of affine invariances (scale, rotational, translational) to image transformations. Convolutional layers are translation equivariant by design, but in their basic form lack invariances. In this work we investigate how best to include rotational invariance in a CNN for image classification. Our experiments show that networks trained with data augmentation alone can classify rotated images nearly as well as in the normal unrotated case; this increase in representational power comes only at the cost of training time. We also compare data augmentation versus two modified CNN models for achieving rotational invariance or equivariance, Spatial Transformer Networks and Group Equivariant CNNs, finding no significant accuracy increase with these specialized methods. In the case of data augmented networks, we also analyze which layers help the network to encode the rotational invariance, which is important for understanding its limitations and how to best retrain a network with data augmentation to achieve invariance to rotation.`

## What can you do with this code

You can train a model on the [MNIST](http://yann.lecun.com/exdb/mnist/) or [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. Two models will be generated for each training; a *rotated* model, for which the dataset's samples were randomly rotated before, and an *unrotated* model, for which they weren't modified at all.

The available models are the [AllConvolutional network](https://arxiv.org/abs/1412.6806) and a [simple Convolutional Network](https://github.com/facundoq/rotational_invariance_data_augmentation/blob/master/pytorch/model/simple_conv.py) with or without a [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025) or with modified [Group Equivariant Convolutional Layers](https://arxiv.org/abs/1602.07576). 

## How to run

These instructions have been tested on a modern ubuntu-based distro with python version>=3.5.  

* Clone the repository and cd to it:
    * `git clone https://github.com/facundoq/rotational_invariance_data_augmentation.git`
    * `cd rotational_invariance_data_augmentation` 
* Create a virtual environment and activate it (requires python3 with the venv module and pip):
    * `python3 -m venv .env`
    * `source .env/bin/activate`
* Install libraries
    * `pip install -r requirements.txt`
* Install [GrouPy with PyTorch support](https://github.com/adambielski/GrouPy)
   * `mkdir ~/dev` (or whenever you want to download GrouPy's repo)
   * `cd ~/dev`
   * `git clone https://github.com/adambielski/GrouPy.git`
   * `cd GrouPy`
   * `python setup.py install` (make sure the venv is active for this step)
   * go back this repo's folder
   
* Run the experiments with `python experiment_name> <model> <dataset>`
    * `experiment_rotation.py` trains two models with the dataset: one with the vanilla version, the other with a data-augmented version via rotations.
    * `experiment_accuracy_vs_rotation.py` evaluates how the model's accuracy varies wrt the rotation of the samples
    * `experiment_retraining.py` (NOT YET AVAILABLE) Evaluates the accuracy of a model trained on a unrotated dataset after retraining some parts of it. 
* The folder `plots` contains the results for any given model/dataset combination

