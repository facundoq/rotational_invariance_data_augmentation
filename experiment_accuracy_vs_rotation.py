## NOTE:
## You should run "experiment_rotation.py" before this script to generate the models for
## a given dataset/model combination

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging
logging.getLogger().setLevel(logging.DEBUG)

from pytorch import dataset as datasets
import torch
import pytorch.experiment.utils as utils

model_name,dataset_name=utils.parse_model_and_dataset("Accuracy vs rotation Experiment.")
# model_name=pytorch_models.AllConv.__name__
# model_name=pytorch_models.SimpleConv.__name__
# dataset_name="mnist"


print(f"### Loading dataset {dataset_name} and model {model_name}....")
verbose=False

use_cuda=torch.cuda.is_available()

dataset = datasets.get_dataset(dataset_name)
if verbose:
    print(dataset.summary())


from pytorch.experiment import rotation
model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)
if verbose:
    print("### ", model)
    print("### ", rotated_model)
    print("### Scores obtained:")
    rotation.print_scores(scores)

from pytorch.experiment import invariance_evaluation

n_rotations=16
#results=invariance_evaluation.run(model,dataset,config)
rotated_results,classes,rotations=invariance_evaluation.run(rotated_model,dataset,config,n_rotations)
results,classes,rotations=invariance_evaluation.run(model,dataset,config,n_rotations)


base_folder="plots/accuracy_rotation"
rotated_fig=invariance_evaluation.plot_results(rotated_results,classes,rotations)
name=f"{base_folder}/{dataset.name}_{model.name}_rotated.png"
rotated_fig.savefig(name)

fig=invariance_evaluation.plot_results(results,classes,rotations)
name=f"{base_folder}/{dataset.name}_{model.name}_unrotated.png"
fig.savefig(name)
