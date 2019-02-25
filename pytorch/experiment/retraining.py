from utils import autolabel
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np

from pytorch.training import test,train,eval_scores

def retraining_accuracy_barchart(model, dataset, unrotated_accuracies,rotated_accuracies,labels,savefig=True):
    import os

    assert len(unrotated_accuracies) == len(rotated_accuracies) == len(labels)
    unrotated_accuracies=np.array(unrotated_accuracies)
    rotated_accuracies = np.array(rotated_accuracies)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(20,8),dpi=150)


    bar_width = 0.2
    index = np.arange(n) - np.arange(n)*bar_width*2.5


    opacity = 0.4
    rects1 = ax.bar(index, unrotated_accuracies, bar_width,
                    alpha=opacity, color='b',
                    label="Test unrotated")

    rects2 = ax.bar(index + bar_width, rotated_accuracies, bar_width,
                    alpha=opacity, color='r',
                    label="Test rotated")
    fontsize = 15
    ax.set_ylim(0, 1.19)
    ax.set_xlabel('Layers retrained',fontsize=fontsize+2)
    ax.set_ylabel('Test accuracy',fontsize=fontsize+2)
    ax.set_title(f'Accuracy on test sets after retraining for {model} on {dataset}.',fontsize=fontsize+4)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels,fontsize=fontsize)
    ax.legend(loc="upper center",fontsize=fontsize+2)

    autolabel(rects1, ax,fontsize=fontsize)
    autolabel(rects2, ax,fontsize=fontsize)
    fig.tight_layout()
    path=os.path.join("plots/", f"retraining_{model}_{dataset}.png")
    if savefig:
        plt.savefig(path)
    plt.show()
    return path



RetrainConfig = namedtuple('RetrainConfig', 'batch_size initial_epochs retrain_epochs use_cuda '
                                            'loss_function')

def retraining(model_optimizer_generator,retrained_layers_schemes,config,dataset):
    train_dataset, rotated_train_dataset = get_data_generator(dataset.x_train, dataset.y_train, config.batch_size)
    test_dataset, rotated_test_dataset = get_data_generator(dataset.x_test, dataset.y_test, config.batch_size)

    model,optimizer=model_optimizer_generator()

    print("Training vanilla network with unrotated dataset..")
    history = train(model, config.initial_epochs, optimizer, config.use_cuda, train_dataset, test_dataset,
                    config.loss_function)

    _, accuracy, _, _ = test(model, test_dataset, config.use_cuda, config.loss_function)
    _, rotated_accuracy, _, _ = test(model, rotated_test_dataset, config.use_cuda, config.loss_function)
    unrotated_accuracies=[accuracy]
    rotated_accuracies = [rotated_accuracy]



    models={"None":model}
    for retrained_layers in retrained_layers_schemes:
        retrained_model,retrained_model_optimizer=model_optimizer_generator(previous_model=model,trainable_layers=retrained_layers)

        # freeze_layers_except(retrained_model.layers(),retrained_model.layer_names(),retrained_layers)

        #for name, val in retrained_model.named_parameters():
         #   print(name, val.requires_grad)

        retrained_layers_id="_".join(retrained_layers)
        print(f"Retraining {retrained_layers} with rotated dataset:")
        history=train(retrained_model,config.retrain_epochs,retrained_model_optimizer,config.use_cuda,rotated_train_dataset,
              rotated_test_dataset,config.loss_function)
        models["retrained_"+retrained_layers_id]=retrained_model
        _, accuracy, _, _ = test(retrained_model, test_dataset, config.use_cuda, config.loss_function)
        _, rotated_accuracy, _, _ = test(retrained_model, rotated_test_dataset, config.use_cuda, config.loss_function)
        unrotated_accuracies.append(accuracy)
        rotated_accuracies.append(rotated_accuracy)


    datasets = {"test_dataset": test_dataset, "rotated_test_dataset": rotated_test_dataset,
                "train_dataset": train_dataset, "rotated_train_dataset": rotated_train_dataset}
    print("Evaluating accuracy for all models/datasets:")
    scores = eval_scores(models, datasets, config, config.loss_function)


    return scores,models,unrotated_accuracies,rotated_accuracies



def freeze_layers_except(layers,layer_names,layers_to_train):
    for i in range(len(layers)):
        name=layer_names[i]
        layer=layers[i]
        requires_grad=name in layers_to_train
        #print(f"Layer {name}: setting requires_grad to {requires_grad}.")
        for param in layer.parameters():
            param.requires_grad=requires_grad
