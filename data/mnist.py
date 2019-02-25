"""MNIST handwritten digits dataset.
"""

from . import util

import numpy as np
import os
import logging

def load_data(path):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path=os.path.join(path,"mnist")
    if not os.path.exists(path):
        os.mkdir(path)

    filename='mnist.npz'
    filepath=os.path.join(path,filename)
    url="https://s3.amazonaws.com/img-datasets/mnist.npz"
    if not os.path.exists(filepath):
        logging.warning(f"Downloading mnist from {url} to {filepath}")
        util.download_file(url,filepath)

    f = np.load(filepath)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
    input_shape=(28,28,1)
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return (x_train, y_train), (x_test, y_test), input_shape, labels


