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

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 1. Data analysis

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
def show_data(data, label):
    print label
    print data.head().to_string()
    print data.describe().to_string()


def plot_data(data):
    train['Died'] = 1 - train['Survived']
    train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                               stacked=True, color=['g', 'r'])
    plt.show()

    data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                               stacked=True, color=['g', 'r'])
    plt.show()

    sns.violinplot(x='Sex', y='Age',
                   hue='Survived', data=data,
                   split=True,
                   palette={0: "r", 1: "g"}
                   )
    plt.show()

    plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']],
             stacked=True, color = ['g','r'],
             bins = 50, label = ['Survived','Dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend()
    plt.show()

    ax = plt.subplot()
    ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'],
               c='green', s=data[data['Survived'] == 1]['Fare'])
    ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'],
               c='red', s=data[data['Survived'] == 0]['Fare'])
    plt.show()

    # fare - class correlation
    ax = plt.subplot()
    ax.set_ylabel('Average fare')
    data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax)
    plt.show()


    sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data, split=True, palette={0: "r", 1: "g"})
    plt.show()


show_data(train, 'train set:')
show_data(test, 'test set: ')
# plot_data(train)


# 2. Feature engineering
def status(feature):
    print 'Processing', feature, ': ok'


def get_combined_data():
    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop(['Survived'], 1, inplace=True)

    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)

    return combined

combined = get_combined_data()

