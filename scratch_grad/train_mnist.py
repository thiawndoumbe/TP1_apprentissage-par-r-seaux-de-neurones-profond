import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from scratch_grad.optimizers import SGD
from scratch_grad.variable import Variable

if __name__ == '__main__':
    # Random seedz
    np.random.seed(42)
    random.seed(42)

    # Training parameters
    num_epochs = 30
    learning_rate = 1e-3

    # Network
    # TODO Creer un réseau à 3 couches linéaires
    # w_layer_1_data = np.random.randn(784, 784).astype(np.float32)  # Poids pour la première couche (784 -> 128)
    # b_layer_1_data = np.random.randn(1, 784).astype(np.float32)  # Biais pour la première couche

    # w_layer_2_data = np.random.randn(784, 784).astype(np.float32)  # Poids pour la deuxième couche (128 -> 128)
    # b_layer_2_data = np.random.randn(1, 784).astype(np.float32)  # Biais pour la deuxième couche

    # w_layer_3_data = np.random.randn(784, 10).astype(np.float32)  # Poids pour la troisième couche (128 -> 10)
    # b_layer_3_data = np.random.randn(1, 10).astype(np.float32)  # Biais pour la troisième couche
    
    w_layer_1_data = np.random.randn(784, 784).astype(np.float32) * 0.01
    b_layer_1_data = np.zeros((1, 784), dtype=np.float32)

    w_layer_2_data = np.random.randn(784, 784).astype(np.float32) * 0.01
    b_layer_2_data = np.zeros((1, 784), dtype=np.float32)

    w_layer_3_data = np.random.randn(784, 10).astype(np.float32) * 0.01
    b_layer_3_data = np.zeros((1, 10), dtype=np.float32)


    # Initialisation des variables
    w_layer_1 = Variable(w_layer_1_data, name='w_layer_1')
    b_layer_1 = Variable(b_layer_1_data, name='b_layer_1')
    w_layer_2 = Variable(w_layer_2_data, name='w_layer_2')
    b_layer_2 = Variable(b_layer_2_data, name='b_layer_2')
    w_layer_3 = Variable(w_layer_3_data, name='w_layer_3')
    b_layer_3 = Variable(b_layer_3_data, name='b_layer_3')

    # Paramètres du résea
    params = [w_layer_1, b_layer_1, w_layer_2, b_layer_2, w_layer_3, b_layer_3]



    # Optimizer
    optimizer = SGD(params, lr=learning_rate)

    # Dataset
    mnist = MNIST(root='data/', train=True, download=True)
    mnist.transform = ToTensor()

    mnist_test = MNIST(root='data/', train=False, download=True)
    mnist_test.transform = ToTensor()

    # Only take a small subset of MNIST
    mnist = Subset(mnist, range(len(mnist) // 16))
    mnist_test = Subset(mnist_test, range(32))

    # Dataloaders
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True)
    val_loader = DataLoader(mnist_test, batch_size=1)

    # Logging
    train_loss_by_epoch = []
    train_acc_by_epoch = []
    val_loss_by_epoch = []
    val_acc_by_epoch = []

    epochs = list(range(num_epochs))
    for epoch in epochs:
        train_predictions = []
        train_losses = []
        # Training loop
        for x, y in tqdm.tqdm(train_loader):
            # Put the data into Variables
            x = x.numpy().reshape((1, 784))
            x = Variable(np.array(x), name='x')
            y = y.numpy().reshape((1, 1))
            y = Variable(np.array(y), name='y')

            # TODO Passer les données dans le réseau avec activations ReLU
            # Calcul de la première couche
            a_1 = x @ w_layer_1 +  b_layer_1 
            a_1._name = 'a1'
            z_1 = a_1.relu()
            z_1._name = 'z1'

            a_2 = z_1 @ w_layer_2  + b_layer_2
            a_2._name = 'a2'
            z_2 = a_2.relu()
            z_2._name = 'z2'

            z_3 = z_2@  w_layer_3 +  b_layer_3
            z_3._name = 'z3'
            # TODO Calculer la perte
            loss = z_3.cross_entropy(y)
            
            display_graph = False

            if display_graph == True:
              loss.show()
            
            display_graph = False

            # TODO Calculer les gradients et mettre à jour les paramètres
            loss.backward()
            w_layer_3.data = w_layer_3.data - (learning_rate * w_layer_3.grad)
            w_layer_2.data = w_layer_2.data - (learning_rate * w_layer_2.grad)
            w_layer_1.data = w_layer_1.data - (learning_rate * w_layer_1.grad)
            b_layer_3.data = b_layer_3.data - (learning_rate * b_layer_3.grad)
            b_layer_2.data = b_layer_2.data - (learning_rate * b_layer_2.grad)
            b_layer_1.data = b_layer_1.data - (learning_rate * b_layer_1.grad)

            w_layer_3.zero_grad()
            w_layer_2.zero_grad()
            w_layer_1.zero_grad()

            b_layer_3.zero_grad()
            b_layer_2.zero_grad()
            b_layer_1.zero_grad()

            a_1.zero_grad()
            a_2.zero_grad()

            z_1.zero_grad()
            z_2.zero_grad()
            z_3.zero_grad()

            # Logging
            train_losses.append(loss.data)


            # Logging
            train_losses.append(loss.data)
            train_predictions.append(np.argmax(z_3.data, axis=1) == y.data)

        # Validation loop
        val_results = []
        val_losses = []
        for x, y in val_loader:
            # Put the data into Variables
            x = x.numpy().reshape((1, 784))
            x = Variable(np.array(x), name='x')
            y = y.numpy().reshape((1, 1))
            y = Variable(np.array(y), name='y')

            # TODO Passer les données dans le réseau
            a_1 = x @ w_layer_1 + b_layer_1
            z_1 = a_1.relu()
            a_2 = z_1 @ w_layer_2 + b_layer_2
            z_2 = a_2.relu()
            z_3 = z_2 @ w_layer_3 + b_layer_3


            # TODO Calculer la perte
            loss = z_3.cross_entropy(y)


            # Logging
            val_losses.append(loss.data)
            val_results.append(np.argmax(z_3.data, axis=1) == y.data)

        # Compute epoch statistics
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_predictions)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_results)

        # Show progress
        print(f'Epoch {epoch}')
        print(f'\tTrain:\t\tLoss {train_loss},\tAcc {train_acc}')
        print(f'\tValidation:\tLoss {val_loss},\tAcc {val_acc}')

        # Logging
        train_loss_by_epoch.append(train_loss)
        train_acc_by_epoch.append(train_acc)
        val_loss_by_epoch.append(val_loss)
        val_acc_by_epoch.append(val_acc)

    # Draw the accuracy-loss plot
    _, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_ylabel('Accuracy')
    axes[0].plot(epochs, train_acc_by_epoch, label='Train')
    axes[0].plot(epochs, val_acc_by_epoch, label='Validation')
    axes[0].legend()

    axes[1].set_ylabel('Loss')
    axes[1].plot(epochs, train_loss_by_epoch, label='Train')
    axes[1].plot(epochs, val_loss_by_epoch, label='Validation')

    axes[1].set_xlabel('Epochs')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()




