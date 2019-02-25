import requests
import shutil
def download_file(url,filepath):
    with requests.get(url, stream=True) as r:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)



from abc import ABC,abstractmethod
from urllib.parse import urlparse
import requests
import os
import logging
class DatasetLoader(ABC):

    def __init__(self,name):
        self.name=name

    @property
    @abstractmethod
    def urls(self):
        pass

    @abstractmethod
    def preprocess(self,path):
        pass

    @abstractmethod
    def load(self,path):
        pass

    def download(self,path):
        logging.warning(f"Downloading files for {self.name}")
        for url in self.urls:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url)
            filepath = os.path.join(path,filename)
            logging.warning(f"Downloading {url} to {filepath}")
            download_file(url,filepath)
        self.set_downloaded(path)


    def set_status_flag(self, path, status,value=True):
        status_path=os.path.join(path,status)
        if not os.path.exists(status_path):
            open(status_path, 'a').close()
        else:
            raise ValueError(f"Status {status} already set on {path}")

    def get_status_flag(self, path, status):
        status_path = os.path.join(path, status)
        return os.path.exists(status_path)

    def set_downloaded(self,path):
        self.set_status_flag(self, path, "downloaded")

    def get_downloaded_flag(self, path):
        self.get_status_flag(self, path, "downloaded")

    def set_preprocessed_flag(self,path):
        self.set_status_flag(self, path, "preprocessed")

    def get_preprocessed_flag(self, path):
        self.get_status_flag(self, path, "preprocessed")

    def get(self,path,**kwargs):
        path=os.path.join(self.name)
        if not os.path.exists(path):
            os.mkdir(path)
        if not self.get_downloaded_flag(path):
            self.download(path)
        if not self.get_preprocessed_flag(path):
            logging.warning(f"Preprocessing {self.name}...")
            self.preprocess(path)
            logging.warning("Done")
        return self.load(path)

    def download_file(self, url, filepath):
        with requests.get(url, stream=True) as r:
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)


import numpy as np
def split_data(x,y,subject,test_subjects):
    if test_subjects=="subject_dependent":
        x_test=x[::2,:,:,:]
        x_train = x[1::2, :, :, :]
        y_test=y[::2]
        y_train= y[1::2]
        subject_test=subject[::2]
        subject_train = subject[1::2]
    else:
        test_subjects=np.array(test_subjects)
        test_indices=np.isin(subject,test_subjects)
        train_indices=np.logical_not(test_indices)
        x_train=x[train_indices,:,:,:]
        x_test=x[test_indices,:,:,:]
        y_train=y[train_indices]
        y_test=y[test_indices]
        subject_train=subject[train_indices]
        subject_test=subject[test_indices]

    return x_train, x_test, y_train, y_test, subject_train,subject_test