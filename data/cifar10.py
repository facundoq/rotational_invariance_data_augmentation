"""Fashion-MNIST dataset.
"""
import tarfile
import os
import numpy as np
import sys
from six.moves import cPickle
from . import util


def load_data(path):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10'
    path = os.path.join(path,dirname)
    filename = "cifar10.tar.gz"
    filepath = os.path.join(path,filename)
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(filepath):
        util.download_file(origin,filepath)
    if not os.path.exists(os.path.join(path,"cifar-10-batches-py")):
        with tarfile.open(filepath, "r:gz") as tar_ref:
            tar_ref.extractall(path)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')
    batches_path=os.path.join(path,"cifar-10-batches-py")
    for i in range(1, 6):
        fpath = os.path.join(batches_path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(batches_path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    x_train, x_test = x_train.transpose([0, 2, 3, 1]), x_test.transpose([0, 2, 3, 1])

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    input_shape = (32, 32, 3)
    labels = ['dog', 'horse', 'frog', 'airplane', 'cat', 'ship', 'deer', 'bird', 'truck', 'automobile']
    return (x_train, y_train), (x_test, y_test), input_shape,labels

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels