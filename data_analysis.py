import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
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

# 1. Data analysis


# PassengerId: and id given to each traveler on the boat
# Pclass: the passenger class. It has three possible values: 1,2,3 (first, second and third class)
# The Name of the passenger
# The Sex
# The Age
# SibSp: number of siblings and spouses traveling with the passenger
# Parch: number of parents and children traveling with the passenger
# The ticket number
# The ticket Fare
# The cabin number
# The embarkation. This describe three possible areas of the Titanic from which the people embark.
# Three possible values S,C,Q

def show_data(data, label=''):
    print(label)
    print(data.head().to_string())
    print(data.describe().to_string())
    print(data.shape)


def analyze_training_data(train):
    train['Died'] = 1 - train['Survived']
    train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                                   stacked=True, color=['g', 'r'])
    plt.show()

    train.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                                    stacked=True, color=['g', 'r'])
    plt.show()

    sns.violinplot(x='Sex', y='Age',
                   hue='Survived', data=train,
                   split=True,
                   palette={0: "r", 1: "g"}
                   )
    plt.show()

    plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']],
             stacked=True, color=['g', 'r'],
             bins=50, label=['Survived', 'Dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend()
    plt.show()

    ax = plt.subplot()
    ax.scatter(train[train['Survived'] == 1]['Age'], train[train['Survived'] == 1]['Fare'],
               c='green', s=train[train['Survived'] == 1]['Fare'])
    ax.scatter(train[train['Survived'] == 0]['Age'], train[train['Survived'] == 0]['Fare'],
               c='red', s=train[train['Survived'] == 0]['Fare'])
    plt.show()

    # fare - class correlation
    ax = plt.subplot()
    ax.set_ylabel('Average fare')
    train.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax)
    plt.show()

    sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=train, split=True, palette={0: "r", 1: "g"})
    plt.show()
