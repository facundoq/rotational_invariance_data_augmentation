import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch

from pytorch.dataset import ImageDataset
from pytorch import training

def run(model,dataset,config,n_rotations,batch_size=256):
    x = dataset.x_test
    y=dataset.y_test
    y_ids=y.argmax(axis=1)
    classes=np.unique(y_ids)
    classes.sort()
    rotations=np.linspace(-180,180,n_rotations,endpoint=False)
    mean_accuracies=np.zeros((dataset.num_classes,len(rotations)))
    for i, c in enumerate(classes):
        # logging.debug(f"Evaluating invariances for class {c}...")
        ids=np.where(y_ids==c)
        ids=ids[0]
        x_class,y_class=x[ids,:],y[ids]
        class_dataset=ImageDataset(x_class,y_class,rotation=True)
        class_mean_accuracies= evaluate_class(class_dataset,model,config,rotations,batch_size)
        mean_accuracies[i,:]=class_mean_accuracies
    return mean_accuracies,classes,rotations

# train_dataset, rotated_train_dataset = get_data_generator(dataset.x_train, dataset.y_train, config.batch_size)
# test_dataset, rotated_test_dataset = get_data_generator(dataset.x_test, dataset.y_test, config.batch_size)

#returns a list of RunningMeanAndVariance objects,
# one for each intermediate output of the model.
#Each RunningMeanAndVariance contains the mean and std of each intermediate
# output over the set of rotations
def evaluate_class(dataset,model,config,rotations,batch_size):

    dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)


    mean_accuracies=[]
    for i,r in enumerate(rotations):
        degrees = (r - 1, r + 1)
        # logging.debug(f"    Rotation {degrees}...")
        dataset.update_rotation_angle(degrees)
        loss, accuracy, correct, n=training.test(model, dataloader,config.use_cuda,torch.nn.NLLLoss())
        mean_accuracies.append(accuracy)
    return np.array(mean_accuracies)



def plot_results(mean_accuracies,classes,rotations):
    f,ax=plt.subplots(1,1,dpi=200)
    im=ax.imshow(mean_accuracies,vmin=0,vmax=1,cmap="inferno")
    #ax.axis("off")
    ax.set_ylabel("class")
    ax.set_yticks(classes)
    ax.set_xlabel("rotation angle (º)")
    ax.set_xticks(np.arange(len(rotations)))
    ax.set_xticklabels(np.round(rotations),fontsize=6)

    f.colorbar(im)
    plt.show()
    return f

