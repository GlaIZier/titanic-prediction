import pandas as pd

from matplotlib import pyplot as plt
import pylab as plot


pd.options.display.max_columns = 100
params = {
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [25, 7]
}
plot.rcParams.update(params)


def plot_train_val_loss(hist, points=500):
    history_dict = hist.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    points = range(0, points)
    plt.plot(points, loss_values, 'bo', label='Training loss')
    plt.plot(points, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_train_val_acc(hist, points=500):
    history_dict = hist.history
    loss_values = history_dict['acc']
    val_loss_values = history_dict['val_acc']
    points = range(0, points)
    plt.plot(points, loss_values, 'bo', label='Training acc')
    plt.plot(points, val_loss_values, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
