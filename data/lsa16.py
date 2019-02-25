
from . import util
import logging
import numpy as np
import os

from skimage import io

from skimage import transform
import zipfile

LSA16_w = 32
LSA16_h = 32
LSA16_class = 16


def load_images(path_images):
    # tensor con toda la BD
    files = sorted(os.listdir(path_images))
    # base, ext = os.path.splitext()
    # if ext.endswith("jpg") or ext.endswith("jpeg") or ext.endswith("png"):
    files=list(filter(lambda f: os.path.splitext(f)[1].endswith("jpg")
                 or os.path.splitext(f)[1].endswith("png")
                 or os.path.splitext(f)[1].endswith("jpeg")
                 ,files))
    n = len(files)
    x = np.zeros((n, LSA16_w, LSA16_h, 3),dtype='uint8')
    y = np.zeros(n,dtype='uint8')
    subjects = np.zeros(n)

    # cargar imagenes con labels
    for (i, filename) in enumerate(files):

            # cargar la imagen actual
            image = io.imread(os.path.join(path_images, filename))

            #image = color.rgb2gray(image)
            image = transform.resize(image, (LSA16_w, LSA16_h),mode="reflect",preserve_range=True,anti_aliasing=True)

            x[i, :, :, :] = image
            # obtener label de la imagen, en base al primer dígito en el nombre de archivo
            y[i] = int(filename.split("_")[0]) - 1
            subjects[i] = int(filename.split("_")[1]) - 1
    return x, y, subjects

def download_and_extract(version, folderpath,images_folderpath):
    if not os.path.exists(folderpath):
        logging.warning(f"Creating folders {folderpath}  and {images_folderpath}")
        os.mkdir(folderpath)
        os.mkdir(images_folderpath)
        print(images_folderpath)

    filename = version + ".zip"
    zip_filepath = os.path.join(folderpath, filename)
    if not os.path.exists(zip_filepath):
        print("Downloading lsa16 version=%s to folder %s ..." % (version, zip_filepath))
        base_url = "http://facundoq.github.io/unlp/lsa16/data/"
        origin = base_url + filename
        util.download_file(origin, zip_filepath)
    if not os.listdir(images_folderpath):
        print("Extracting images..." )
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(images_folderpath)


def load_data(path,version="lsa32x32_nr_rgb_black_background", test_subjects=[9]):
    path=os.path.join(path,f"lsa16_{version}")
    images_folderpath=os.path.join(path,"images")

    # download dataset (if necessary)
    download_and_extract(version, path,images_folderpath)
    print("Loading images from %s" % images_folderpath)

    # load images
    x, y, subjects = load_images(images_folderpath)

    # split in train/test datasets
    test_indices = np.isin(subjects, test_subjects)
    train_indices = np.invert(test_indices)
    x_test = x[test_indices, :, :, :]
    y_test = y[test_indices]
    x_train = x[train_indices, :, :, :]
    y_train = y[train_indices]

    input_shape=(32,32,3)
    labels = ["five", "four", "horns", "curve", "fingers together", "double", "hook", "index", "l", "flat hand","mitten", "beak", "thumb", "fist", "telephone", "V"]

    return x_train, x_test, y_train, y_test, input_shape,labels