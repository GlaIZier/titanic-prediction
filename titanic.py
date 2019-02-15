import tkinter as tk
import numpy as np
import pandas as pd
pd.options.display.max_columns = 100

import matplotlib
from matplotlib import pyplot as plt

import seaborn as sns

import pylab as plot
params = {
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [25, 7]
}
plot.rcParams.update(params)


# PassengerId: and id given to each traveler on the boat
# Pclass: the passenger class. It has three possible values: 1,2,3 (first, second and third class)
# The Name of the passeger
# The Sex
# The Age
# SibSp: number of siblings and spouses traveling with the passenger
# Parch: number of parents and children traveling with the passenger
# The ticket number
# The ticket Fare
# The cabin number
# The embarkation. This describe three possible areas of the Titanic from which the people embark.
# Three possible values S,C,Q
def show_data(data):
    print(data.head().to_string())
    print(data.describe().to_string())

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
show_data(train)
show_data(test)
train['Died'] = 1 - train['Survived']
train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                          stacked=True, color=['g', 'r'])
plt.show()

