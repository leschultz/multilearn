from matplotlib import pyplot as pl
from sklearn import metrics

import numpy as np

import matplotlib
import json
import os

# Font styles
font = {'font.size': 16, 'lines.markersize': 10}
matplotlib.rcParams.update(font)


def plot_dump(data, fig, ax, save, legend=True):
    '''
    Function to dump figures.

    Args:
        data (dict): Data to dump in json file.
        fig (object): Figure object.
        ax (object): Axes object.
        save (str): The location to save plot.
    '''

    fig.tight_layout()

    if legend:

        fig_legend, ax_legend = pl.subplots()
        ax_legend.axis(False)

        legend = ax_legend.legend(
                                  *ax.get_legend_handles_labels(),
                                  frameon=False,
                                  loc='center',
                                  bbox_to_anchor=(0.5, 0.5)
                                  )

        ax_legend.spines['top'].set_visible(False)
        ax_legend.spines['bottom'].set_visible(False)
        ax_legend.spines['left'].set_visible(False)
        ax_legend.spines['right'].set_visible(False)

        fig_legend.savefig(save+'_legend.png', bbox_inches='tight', dpi=400)

        ax.legend([]).set_visible(False)

        pl.close(fig_legend)

    fig.savefig(save+'.png', bbox_inches='tight', dpi=400)

    pl.close(fig)

    with open(save+'.json', 'w') as handle:
        json.dump(data, handle)


def parity(y, y_pred, sigma_y, save, color):

    '''
    Make a parity plot.

    Args:
        y (np.ndarray): The true target variable.
        y_pred (np.ndarray): The predicted target variable.
        sigma_y (float): The standard deviation of y.
        save (str): The directory to save plot.
        color (str): The color of the plot.
    '''

    rmse = metrics.mean_squared_error(y, y_pred)**0.5

    if y.shape[0] > 1:
        rmse_sigma = rmse/sigma_y
    else:
        rmse_sigma = np.nan

    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    label = r'$RMSE/\sigma_{y}=$'
    label += r'{:.2}'.format(rmse_sigma)
    label += '\n'
    label += r'$RMSE=$'
    label += r'{:.2}'.format(rmse)
    label += '\n'
    label += r'$MAE=$'
    label += r'{:.2}'.format(mae)
    label += '\n'
    label += r'$R^{2}=$'
    label += r'{:.2}'.format(r2)

    fig, ax = pl.subplots()

    ax.scatter(
               y,
               y_pred,
               marker='.',
               zorder=2,
               color=color,
               label=label,
               )

    limits = []
    min_range = min(min(y), min(y_pred))
    max_range = max(max(y), max(y_pred))
    span = max_range-min_range
    limits.append(min_range-0.1*span)
    limits.append(max_range+0.1*span)

    # Line of best fit
    ax.plot(
            limits,
            limits,
            label=r'$y=\hat{y}$',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_ylabel(r'$\hat{y}$')
    ax.set_xlabel('y')

    h = 8
    w = 8

    fig.set_size_inches(h, w, forward=True)

    data = {}
    data[r'$RMSE$'] = float(rmse)
    data[r'$RMSE/\sigma_{y}$'] = float(rmse_sigma)
    data[r'$MAE$'] = float(mae)
    data[r'$R^{2}$'] = float(r2)
    data['y'] = y.tolist()
    data['y_pred'] = y_pred.tolist()

    plot_dump(data, fig, ax, save)


def generate(
             df_parity,
             df_loss,
             save='.',
             ):

    '''
    Generate both parity and learning curve plots.

    Args:
        df_parity (pd.DataFrame): Parity plot data.
        df_loss (pd.DataFrame): Learning curve data.
        save (str): Location to save all outputs.
    '''

    for group, values in df_parity.groupby(['data', 'split']):

        y = values['y']
        sigma_y = y.std()
        y = y.values
        y_pred = values['p'].values

        data_indx, data_set = group

        if data_set == 'train':
            color = 'g'
        elif data_set == 'val':
            color = 'b'
        elif data_set == 'test':
            color = 'r'

        save_dir = os.path.join(*[save, f'{data_indx}', 'parity'])
        os.makedirs(save_dir, exist_ok=True)
        newsave = os.path.join(save_dir, '{}.png'.format(data_set))

        parity(y, y_pred, sigma_y, newsave, color)

    for group, values in df_loss.groupby(['data', 'split']):

        x = values['epoch'].values
        y = values['loss'].values

        data_indx, data_set = group

        if data_set == 'train':
            color = 'g'
        elif data_set == 'val':
            color = 'b'
        elif data_set == 'test':
            color = 'r'

        save_dir = os.path.join(*[save, f'{data_indx}', 'loss_vs_epoch'])
        os.makedirs(save_dir, exist_ok=True)
        newsave = os.path.join(save_dir, '{}.png'.format(data_set))

        learning_curve(x, y, newsave, data_set, color)


def learning_curve(x, y, save, group, color):
    '''
    Plot the loss versus the epoch.

    Args:
        x (list): The epochs.
        y (list): The loss.
        save (str): The save location.
        group (str): The data set in question.
        color (str): The plot color.
    '''

    # Regular plot
    fig, ax = pl.subplots()

    val = min(y)

    label = '{}: lowest loss value: {:.2f}'.format(group.capitalize(), val)
    label += '\n'
    label += '{}: last loss value: {:.2f}'.format(group.capitalize(), y[-1])

    ax.plot(
            x,
            y,
            marker='.',
            color=color,
            label=label,
            )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    data = {}
    data['mae'] = y.tolist()
    data['epoch'] = x.tolist()

    plot_dump(data, fig, ax, save, False)
