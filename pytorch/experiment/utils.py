import numpy as np

class RunningMeanAndVariance:

    def __repr__(self):

        return f"RunningMeanAndVariance(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def var(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return np.sqrt(self.var())


import argparse,argcomplete
import pytorch.experiment.model_loading as models
import datasets

def parse_model_and_dataset(description):

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('model', metavar='m',
                        help=f'Model to train/use. Allowed values: {", ".join(models.get_model_names())}'
                        , choices=models.get_model_names())
    parser.add_argument('dataset', metavar='d',
                        help=f'Dataset to train/eval model. Allowed values: {", ".join(datasets.names)}'
                        ,choices=datasets.names)
    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    return args.model,args.dataset
