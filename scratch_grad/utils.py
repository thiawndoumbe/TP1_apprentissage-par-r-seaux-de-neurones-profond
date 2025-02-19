import numpy as np
from matplotlib import pyplot as plt

from scratch_grad.variable import Variable


def show_2d_function(fct, min_val=-5, max_val=5, mesh_step=.01, *, optimal=None, bar=True, ax=None, **kwargs):
    w1_values = np.arange(min_val, max_val + mesh_step, mesh_step)
    w2_values = np.arange(min_val, max_val + mesh_step, mesh_step)

    w2, w1 = np.meshgrid(w2_values, w1_values)
    w_grid = np.stack((w1.flatten(), w2.flatten()))
    f = fct(Variable(w_grid)).data.copy()
    fct_values = f.reshape(w1_values.shape[0], w2.shape[0]).T

    w1_values, w2_values = w1_values, w2_values

    if ax is not None:
        plt.sca(ax)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'RdBu'
    plt.contour(w1_values, w2_values, fct_values, 40, **kwargs)
    plt.xlim((min_val, max_val))
    plt.ylim((min_val, max_val))
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')

    if bar:
        plt.colorbar()

    if optimal is not None:
        plt.scatter(*optimal, s=200, marker='*', c='r')


def show_2d_trajectory(w_history, fct, min_val=-5, max_val=5, mesh_step=.5, *, optimal=None, ax=None):
    show_2d_function(fct, min_val, max_val, mesh_step, optimal=optimal, ax=ax)

    if len(w_history) > 0:
        trajectory = np.array(w_history)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'o--', c='g')

    plt.title('Trajectoire de la descente en gradient')
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')


def show_learning_curve(loss_list, loss_opt=None, ax=None):
    if ax is not None:
        plt.sca(ax)
    plt.plot(np.arange(1, len(loss_list) + 1), loss_list, 'o--', c='g', label='$F(\\mathbf{w})$')
    if loss_opt is not None:
        plt.plot([1, len(loss_list)], 2 * [loss_opt], '*--', c='r', label='optimal')
    plt.title('Valeurs de la fonction objectif')
    plt.xlabel('Itérations')
    plt.legend()


def show_optimization(w_history, loss_history, fct, optimal=None, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4))
    if title is not None:
        fig.suptitle(title)
    show_2d_trajectory(w_history, fct, optimal=optimal, ax=axes[0])
    show_learning_curve(loss_history, loss_opt=fct(optimal).data, ax=axes[1])
