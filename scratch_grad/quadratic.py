import numpy as np
from matplotlib import pyplot as plt

from scratch_grad.utils import show_optimization
from scratch_grad.variable import Variable

if __name__ == '__main__':
    num_epoch = 20

    x_arr = np.array([[1, 1], [0, -1], [2, 0.5]])
    y_arr = np.array([[-1.0], [3], [2]])
    w_opt = np.linalg.inv(x_arr.T @ x_arr) @ x_arr.T @ y_arr

    x = Variable(x_arr)
    y = Variable(y_arr)


    # Model
    def objective_function(w: Variable):
        error = x @ w - y
        return (error * error).mean_row()


    def loss_function(pred, target):
        error = pred - target
        return (error * error).mean_row()


    def show_objective_optimization(name, w_history, loss_history, **kwargs):
        return show_optimization(w_history, loss_history, objective_function, optimal=w_opt, title=name, **kwargs)


    def optimize_simple(nb_iter, lr):
        # TODO Créer un neurone avec des poids de [0.5, 0.5]
        neuron =  Variable(np.array([[0.5], [0.5]]))
        w_history = []
        loss_history = []
        for t in range(nb_iter):
            # TODO Calculer la prédiction du neurone
            pred = x @ neuron

            # TODO Calculer la fonction objectif
            loss = loss_function(pred, y)

            w_history.append(neuron.data.copy())
            loss_history.append(loss.data)

            # TODO Mettre à jour les poids
            loss.backward()
            neuron.data -= lr * neuron.grad  # Mise à jour des poids
            neuron.zero_grad()  # Remise à zéro du gradient

        return w_history, loss_history


    def optimize(nb_iter, optimizer_type, **optimizer_kwargs):
        # TODO Créer un neurone avec des poids de [0.5, 0.5]
        neuron = Variable(np.array([[0.5], [0.5]]))
        optimizer = optimizer_type([neuron], **optimizer_kwargs)
        w_history = []
        loss_history = []
        for t in range(nb_iter):
            # TODO Calculer la prédiction du neurone
            pred = x @ neuron

            # TODO Calculer la fonction objectif
            loss = loss_function(pred, y)

            w_history.append(neuron.data.copy())
            loss_history.append(loss.data)

            # TODO Mettre à jour les poids avec l'optimizer
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return w_history, loss_history


    # No optimizer
    w_history, loss_history = optimize_simple(num_epoch, lr=0.3)
    show_objective_optimization(f'Simple', w_history, loss_history)
    plt.show()

    # lr = 0.3
    # w_history, loss_history = optimize(num_epoch, SGD, lr=lr)
    # show_objective_optimization(f'SGD {lr=}', w_history, loss_history)
    # plt.show()
    
