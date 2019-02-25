import progressbar
import torch
import numpy as np

def print_results(dataset,loss,accuracy,correct,n):
    print('{} => Loss: {:.4f}, Accuracy: {:.2f}% ({}/{})'.format(dataset,
        loss, 100. * accuracy, correct, n),flush=True)

def train(model,epochs,optimizer,use_cuda,train_dataset,test_dataset,loss_function):
    history={"acc":[],"acc_val":[],"loss":[],"loss_val":[]}
    model.train()
    for epoch in range(1, epochs + 1):
        loss,accuracy,correct,n=train_epoch(model,epoch,optimizer,use_cuda,train_dataset,loss_function)

        #train_results = test(model, train_dataset, use_cuda)
        #print_results("Train",*train_results)

        #loss, accuracy, correct,n= test(model,train_dataset, use_cuda, loss_function)
        test_results = test(model,test_dataset,use_cuda,loss_function)
        print_results("Test", *test_results)
        history["loss"].append(loss)
        history["loss_val"].append(test_results[0])
        history["acc"].append(accuracy)
        history["acc_val"].append(test_results[1])
    return history


def test(model, dataset, use_cuda,loss_function):
    with torch.no_grad():
        model.eval()
        loss = 0
        correct = 0

        for data, target in dataset:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.float()
            # data, target = Variable(data,), Variable(target)

            output = model(data)

            loss += loss_function(output,target).item()*data.shape[0]
            #loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    n=len(dataset.dataset)
    loss /= n
    accuracy = float(correct) / float(n)
    return loss,accuracy,correct,n

def train_epoch(model,epoch,optimizer,use_cuda,train_dataset,loss_function):
    widgets = ["Epoch {}: ".format(epoch), progressbar.Percentage()
               ,progressbar.FormatLabel(' (batch %(value)d/%(max_value)d) ')
               ,' ==stats==> ', progressbar.DynamicMessage("loss")
               ,', ',progressbar.DynamicMessage("accuracy")
               ,', ',progressbar.ETA()
               ]
    progress_bar = progressbar.ProgressBar(widgets=widgets, max_value=len(train_dataset)).start()
    batches=len(train_dataset)
    losses=np.zeros(batches)
    accuracies=np.zeros(batches)
    correct=0

    for batch_idx, (data, target) in enumerate(train_dataset):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        #MODEL OUTPUT
        output = model(data)

        loss = loss_function(output, target)
        # loss = F.nll_loss(output, target)

        # UPDATE PARAMETERS
        loss.backward()
        optimizer.step()


        # ESTIMATE BATCH LOSS AND ACCURACY
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        matches = pred.eq(target.data.view_as(pred)).cpu()
        correct += matches.sum()
        accuracies[batch_idx] = matches.float().mean().item()
        losses[batch_idx] = loss.cpu().item()

        # UPDATE UI
        if batch_idx % 20 == 0:
            progress_bar.update(batch_idx+1,loss=losses[:batch_idx+1].mean(),accuracy=accuracies[:batch_idx+1].mean())

    progress_bar.finish()
    return losses.mean(),accuracies.mean(),correct,len(train_dataset.dataset)


def eval_scores(models,datasets,config,loss_function):
    scores = {}
    for model_name in sorted(models):
        m = models[model_name]
        for dataset_name in sorted(datasets):
            dataset = datasets[dataset_name]
            key = model_name + '_' + dataset_name
            #print(f"Evaluating {key}:")
            loss,accuracy,correct,n=test(m,dataset,config.use_cuda,loss_function)

            scores[key] = (loss,accuracy)

    return scores



def add_weight_decay(parameters, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in parameters:
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]
