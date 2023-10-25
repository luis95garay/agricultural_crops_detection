# Import libraries
import torch
import numpy as np

# PyTorch dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch model
import torch.nn as nn
import torch.optim as optim

# Mlflow
import mlflow
import mlflow.pytorch

# Prefect
from prefect import flow, task

from neural_networks import Net


@task(retries=3, retry_delay_seconds=2)
def read_data():
    # Data transform to convert data to a tensor and apply normalization

    # augment train and validation dataset with RandomHorizontalFlip and RandomRotation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # randomly flip and rotate
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                download=True, transform=train_transform)
    test_data = datasets.CIFAR10('data', train=False,
                                download=True, transform=test_transform)
    
    return train_data, test_data


@task
def feature_engineering(train_data, test_data):
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader


@task(log_prints=True)
def train_test_model(train_loader, valid_loader, test_loader):
    batch_size = 20
    
    with mlflow.start_run() as run:
        train_on_gpu = torch.cuda.is_available()
        # Params
        params = {
            "conv1": 16,
            "conv2": 32,
            "conv3": 64,
            "kernel": 3,
            "padding": 1,
            "dropout": 0.05,
            "fc1": 40,
            "n_epochs": 1,
            "criterion": "CrossEntropyLoss",
            "optimizer": "SGD"
        }

        mlflow.log_params(params)
        # create a complete CNN
        model = Net(
            conv1 = params['conv1'],
            conv2 = params['conv2'],
            conv3 = params['conv3'],
            kernel = params['kernel'],
            padding = params['padding'],
            dropout = params['dropout'],
            fc1 = params['fc1']
        )
        if train_on_gpu:
            model.cuda()
        # specify loss function (categorical cross-entropy)
        criterion = nn.CrossEntropyLoss()

        # specify optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # number of epochs to train the model
        n_epochs = params['n_epochs']

        valid_loss_min = np.Inf # track change in validation loss

        for epoch in range(1, n_epochs+1):

            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            ###################
            # train the model #
            ###################
            model.train()
            for data, target in train_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)
                
            ######################    
            # validate the model #
            ######################
            model.eval()
            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
            
            # calculate average losses
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)
                
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), 'models/model_cifar.pt')
                valid_loss_min = valid_loss
            mlflow.pytorch.log_model(model, "models")

        # test
        # track test loss
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        model.eval()
        # iterate over test data
        for data, target in test_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)    
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # average test loss
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))
        mlflow.log_metric("avg_test_loss", test_loss)

        for i in range(10):
            if class_total[i] > 0:
                class_loss = 100 * class_correct[i] / class_total[i]
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], class_loss,
                    np.sum(class_correct[i]), np.sum(class_total[i])))
                mlflow.log_metric(f"test_loss_{classes[i]}", class_loss)
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            test_accuracy,
            np.sum(class_correct), np.sum(class_total)))
        mlflow.log_metric("test_accuracy_overall", test_accuracy)


@flow
def main_flow() -> None:
    TRACKING_SERVER_HOST = "localhost"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("cifar-experiments")

    train_data, test_data = read_data()

    train_loader, valid_loader, test_loader = feature_engineering(train_data, test_data)

    train_test_model(train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main_flow()
