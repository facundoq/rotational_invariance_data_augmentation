
# Revisiting Data Augmentation for Rotational Invariance in Convolutional Neural Networks

This repository contains the code necessary to obtain the experimental results published in the article "Revisiting Data Augmentation for Rotational Invariance in Convolutional Neural Networks". 

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
* Run the experiments with `python experiment_name> <model> <dataset>`
    * `experiment_rotation.py` trains two models with the dataset: one with the vanilla version, the other with a data-augmented version via rotations.
    * `experiment_accuracy_vs_rotation.py` evaluates how the model's accuracy varies wrt the rotation of the samples
    * `experiment_retraining.py` Evaluates how
* The folder `plots` contains the results for any given model/dataset combination

