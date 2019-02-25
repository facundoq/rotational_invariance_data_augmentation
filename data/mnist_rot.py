import numpy as np
import os
#from scipy import ndimage
from skimage import io
from skimage import color
from skimage import transform
import zipfile
from . import util
from os.path import expanduser


# Dataset from: http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations

LSA16_w = 32
LSA16_h = 32
LSA16_class = 16


def load_images(folderpath):
    train = np.load(os.path.join(folderpath, "train_all.npz"))
    test = np.load(os.path.join(folderpath,"test.npz"))

    x_train = train['data']
    y_train = train['labels']
    x_test  = test['data']
    y_test  = test['labels']
    # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    # print(x_train.dtype, y_train.dtype, x_test.dtype, y_test.dtype)
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return x_train,x_test,y_train,y_test

def encode_images(folderpath):
    train_fn = os.path.join(folderpath, 'mnist_all_rotation_normalized_float_train_valid.amat')
    test_fn = os.path.join(folderpath, 'mnist_all_rotation_normalized_float_test.amat')

    train_val = np.loadtxt(train_fn)
    test = np.loadtxt(test_fn)

    train_val_data = train_val[:, :-1].reshape(-1, 28, 28,1)
    train_val_data= np.uint8(train_val_data *255)
    train_val_labels = train_val[:, -1]

    test_data = test[:, :-1].reshape(-1, 28, 28,1)
    test_labels = test[:, -1]
    test_data=np.uint8(test_data*255)

    # print(train_val_data.shape, train_val_labels.shape, test_data.shape, test_labels.shape)

    np.savez(os.path.join(folderpath, 'train_all.npz'), data=train_val_data, labels=train_val_labels)
    np.savez(os.path.join(folderpath, 'train.npz'), data=train_val_data[:10000], labels=train_val_labels[:10000])
    np.savez(os.path.join(folderpath, 'valid.npz'), data=train_val_data[10000:], labels=train_val_labels[10000:])
    np.savez(os.path.join(folderpath, 'test.npz'), data=test_data, labels=test_labels)



def download_and_extract(folderpath):
    filename = "mnist_rotation_new.zip"
    zip_filepath = os.path.join(folderpath, filename)
    if not os.path.exists(zip_filepath):
        print("Downloading mnist rot to %s ..." % zip_filepath)
        origin = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
        util.download_file(origin,zip_filepath)
        print("Extracting dataset matrices..." )
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(folderpath)
            encode_images(folderpath)


def load_data(folderpath):
    folderpath=os.path.join(folderpath,"mnist_rot")
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
        download_and_extract(folderpath)
        encode_images(folderpath)
    x_train,x_test,y_train,y_test=load_images(folderpath)

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    input_shape = x_train.shape[1:]
    return x_train, x_test, y_train, y_test, input_shape, labels