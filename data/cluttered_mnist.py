"""CLUTTERED MNIST digits classification dataset.
"""

from . import util
import numpy as np
import os
import logging

# see https://github.com/skaae/recurrent-spatial-transformer-code/blob/master/MNIST_SEQUENCE/create_mnist_sequence.py
# and https://github.com/MasazI/Spatial_Transformer_Network/blob/master/load_data.py

def load_data(path):
    """Loads the Cluttered MNIST dataset dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_val, y_val), (x_test, y_test)`.
    """
    
    path
    foldername='cluttered-mnist'
    folderpath = os.path.join(path,foldername)
    if not os.path.exists(folderpath):
        logging.warning(f"Creating folder {folderpath}...")
        os.mkdir(folderpath)
    filename = 'cluttered-mnist.npz'
    filepath = os.path.join(folderpath, filename)
    if not os.path.exists(filepath):
#     origin = "https://github.com/skaae/recurrent-spatial-transformer-code/raw/master/mnist_sequence3_sample_8distortions_9x9.npz"
#     origin = 'https://github.com/daviddao/spatial-transformer-tensorflow/raw/master/data/mnist_sequence1_sample_5distortions5x5.npz'
    
        origin="https://s3.amazonaws.com/lasagne/recipes/datasets/mnist_cluttered_60x60_6distortions.npz"
        logging.warning(f"Downloading dataset to {filepath} from {origin}")
        util.download_file(origin, filepath)


    mnist_cluttered = np.load(filepath)

    x_train = mnist_cluttered['x_train']
    y_train = mnist_cluttered['y_train'].argmax(axis=-1)


    x_test = mnist_cluttered['x_test']
    y_test = mnist_cluttered['y_test'].argmax(axis=-1)
    
    x_val = mnist_cluttered['x_valid']
    y_val = mnist_cluttered['y_valid'].argmax(axis=-1)
    
    DIM=60
    
    x_train = x_train.reshape((x_train.shape[0], DIM, DIM, 1))
    x_val = x_val.reshape((x_val.shape[0], DIM, DIM, 1))
    x_test = x_test.reshape((x_test.shape[0], DIM, DIM, 1))
    
    input_shape = 60,60,1
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return (x_train, y_train), (x_test, y_test), input_shape,labels