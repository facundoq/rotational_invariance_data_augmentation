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


from pytorch.experiment import retraining


initial_epochs=4
retrain_epochs=7
conv_layer_names= list(filter(lambda name: name.startswith("c"),layer_names))
fc_layer_names= list(filter(lambda name: name.startswith("fc"),layer_names))
individual_layers= [[l] for l in layer_names]
retrained_layers_schemes=individual_layers + [conv_layer_names,fc_layer_names] +[layer_names]

labels=["none"] +layer_names +["conv","fc","all"]

# retrained_layers_schemes=[conv_layer_names,fc_layer_names]
# labels=["conv","fc"]
print("Retraining schemes:\n")
print("\n".join(map(lambda scheme: "_".join(scheme),retrained_layers_schemes)))

config=pytorch_experiment.RetrainConfig(batch_size,initial_epochs,retrain_epochs,use_cuda,loss_function)
scores,models,unrotated_accuracies,rotated_accuracies=pytorch_experiment.retraining(model_optimizer_generator,retrained_layers_schemes,config,dataset)

results=invariance_evaluation.run(model,dataset,config)
rotated_results,classes,rotations=invariance_evaluation.run(rotated_model,dataset,config,n_rotations)
results,classes,rotations=invariance_evaluation.run(model,dataset,config,n_rotations)

conv_filters = {"mnist": 32, "mnist_rot": 32, "cifar10": 64}
fc_filters = {"mnist": 64, "mnist_rot": 64, "cifar10": 128}


def subset(l, indices):
    return [l[i] for i in indices]


def freeze_layers_except(layers, layer_names, layers_to_train):
    for i in range(len(layers)):
        name = layer_names[i]
        layer = layers[i]
        requires_grad = name in layers_to_train
        # print(f"Layer {name}: setting requires_grad to {requires_grad}.")
        for param in layer.parameters():
            param.requires_grad = requires_grad


def model_optimizer_generator(previous_model=None, trainable_layers=None):
    model = pytorch_models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                      conv_filters=conv_filters[dataset.name], fc_filters=fc_filters[dataset.name])
    if use_cuda:
        model = model.cuda()
    if previous_model:
        model.load_state_dict(previous_model.state_dict())

    if trainable_layers:
        freeze_layers_except(model.layers(), model.layer_names(), trainable_layers)

    parameters = pytorch_experiment.add_weight_decay(model.named_parameters(), 1e-9)
    optimizer = optim.Adam(parameters, lr=0.001)

    return model, optimizer


model, optimizer = model_optimizer_generator()

base_folder="plots/retraining"

rotated_fig=retraining.retraining_accuracy_barchart(model.name,dataset.name,unrotated_accuracies,rotated_accuracies,labels,savefig=True)

name=f"{base_folder}/{dataset.name}_{model.name}_retraining.png"
rotated_fig.savefig(name)


