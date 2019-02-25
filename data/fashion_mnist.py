"""Fashion-MNIST dataset.
"""

import gzip
import os

from . import util
import numpy as np



def load_data(path):
    """Loads the Fashion-MNIST dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = os.path.join(path, 'fashion-mnist')
    if not os.path.exists(path):
        os.makedirs(path)

    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        fname_path=os.path.join(path,fname)
        if not os.path.exists(fname_path):
            util.download_file(base + fname,fname_path)
        paths.append(fname_path)

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
    input_shape = (28, 28, 1)
    labels = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]
    return (x_train, y_train), (x_test, y_test), input_shape, labels


