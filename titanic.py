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
    print(label)
    print(data.head().to_string())
    print(data.describe().to_string())


def analyze_training_data():
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


# show_data(train, 'train set:')
# show_data(test, 'test set: ')
# analyze_training_data()


# 2. Feature engineering
def status(feature):
    print('Processing', feature, ': ok')


# todo do we need a combined dataset?
def get_combined_data():
    # extracting and then removing the targets from the training data
    train.drop(['Survived'], 1, inplace=True)

    # merging train data and test data for future feature engineering
    comb = train.append(test)
    comb.reset_index(inplace=True)
    comb.drop(['index', 'PassengerId'], inplace=True, axis=1)

    return comb


def add_titles(comb):
    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    # we extract the title from each name
    comb['Title'] = comb['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated title
    # we map each title
    comb['Title'] = comb.Title.map(title_dictionary)
    # set a single value to Royalty manually
    comb.loc[comb[comb['Title'].isnull()].index[0], 'Title'] = 'Royalty'
    # comb[comb['Title'].isnull()]['Title'] = 'Royalty'

    status('Title')
    return comb

combined = add_titles(get_combined_data())


# To avoid data leakage from the test set, we fill in missing ages in the train using the train set and we fill in ages
# in the test set using values calculated from the train set as well.
def fill_empty_ages(comb):
    # print('The number of empty ages: ', comb.iloc[:891].Age.isnull().sum())
    # calculate median ages for different categories of passengers
    grouped_train = comb.iloc[:891].groupby(['Sex', 'Pclass', 'Title'])
    grouped_median_train = grouped_train.median().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

    def fill_age(row):
        condition = (
            (grouped_median_train['Sex'] == row['Sex']) &
            (grouped_median_train['Title'] == row['Title']) &
            (grouped_median_train['Pclass'] == row['Pclass'])
        )
        return grouped_median_train[condition]['Age'].values[0]
    # replace age with a median one if it's nan
    comb['Age'] = comb.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('age')
    return comb

combined = fill_empty_ages(combined)


def refine_names(comb):
    # we clean the Name variable
    comb.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(comb['Title'], prefix='Title')
    comb = pd.concat([comb, titles_dummies], axis=1)

    # removing the title variable
    comb.drop('Title', axis=1, inplace=True)

    status('names')
    return comb

combined = refine_names(combined)


def fill_empty_fares(comb):
    # print('The number of empty fares: ', comb.iloc[:891].Fare.isnull().sum())
    # there's one missing fare value - replacing it with the mean.
    comb.Fare.fillna(comb.iloc[:891].Fare.mean(), inplace=True)
    status('fare')
    return comb

combined = fill_empty_fares(combined)


def fill_empty_embarked(comb):
    # print('The number of empty fares: ', comb.iloc[:891].Embarked.isnull().sum())
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    frequent_embarked = comb.iloc[:891].Embarked.mode()[0]
    comb.Embarked.fillna(frequent_embarked, inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(comb['Embarked'], prefix='Embarked')
    comb = pd.concat([comb, embarked_dummies], axis=1)
    comb.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return comb

combined = fill_empty_embarked(combined)


# As we don't have any cabin letter in the test set that is not present in the train set, we can replace the whole set
def encode_cabins(comb):
    # replacing missing cabins with U (for Unknown)
    comb.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    comb['Cabin'] = comb['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(comb['Cabin'], prefix='Cabin')
    comb = pd.concat([comb, cabin_dummies], axis=1)

    comb.drop('Cabin', axis=1, inplace=True)
    status('cabin')
    return comb

combined = encode_cabins(combined)


def encode_sex(comb):
    # mapping string values to numerical one
    comb['Sex'] = comb['Sex'].map({'male': 1, 'female': 0})
    status('Sex')
    return comb

combined = encode_sex(combined)


def encode_pclasses(comb):
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(comb['Pclass'], prefix="Pclass")

    # adding dummy variable
    comb = pd.concat([comb, pclass_dummies], axis=1)

    # removing "Pclass"
    comb.drop('Pclass', axis=1, inplace=True)

    status('Pclass')
    return comb

combined = encode_pclasses(combined)


def encode_tickets(comb):

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def clean_ticket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    # Extracting dummy variables from tickets:
    comb['Ticket'] = comb['Ticket'].map(clean_ticket)
    tickets_dummies = pd.get_dummies(comb['Ticket'], prefix='Ticket')
    comb = pd.concat([comb, tickets_dummies], axis=1)
    comb.drop('Ticket', inplace=True, axis=1)

    status('Ticket')
    return comb

combined = encode_tickets(combined)


# addition of a new feature: Large families are grouped together;
# hence, they are more likely to get rescued than people traveling alone.
def add_family_size(comb):
    # introducing a new feature : the size of families (including the passenger)
    comb['FamilySize'] = comb['Parch'] + comb['SibSp'] + 1

    # introducing other features based on the family size
    comb['Singleton'] = comb['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    comb['SmallFamily'] = comb['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    comb['LargeFamily'] = comb['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')
    return comb

combined = add_family_size(combined)
print(combined.head())

