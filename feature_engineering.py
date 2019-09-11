import numpy as np
import pandas as pd

from keras import models
from keras.layers import Dense

pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# 2. Feature engineering

class Data:
    raw_train = None
    raw_test = None
    train_border_index = None
    validation_border_index = None
    engineered_train = None
    engineered_test = None
    x_train_full = None
    y_train_full = None
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None


def status(feature):
    print('Processing', feature, ': ok')


def combine_data(train, test):
    comb = train.append(test)
    comb['is_test'] = 1
    comb.iloc[:train_border_index, comb.columns.get_loc('is_test')] = 0
    status('Combined')
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


# If age is missing, replace it with a mean value
# To avoid data leakage from the test set, we fill in missing ages in the train using the train set and we fill in ages
# in the test set using values calculated from the train set as well.
def fill_empty_ages(comb):
    # print('The number of empty ages: ', comb.iloc[:train_border_index].Age.isnull().sum())
    # calculate median ages for different categories of passengers
    grouped_train = comb.iloc[:train_border_index].groupby(['Sex', 'Pclass', 'Title'])
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


def fill_empty_fares(comb):
    # print('The number of empty fares: ', comb.iloc[:train_border_index].Fare.isnull().sum())
    # there's one missing fare value - replacing it with the mean.
    comb.Fare.fillna(comb.iloc[:train_border_index].Fare.mean(), inplace=True)
    status('fare')
    return comb


def fill_empty_embarked(comb):
    # print('The number of empty fares: ', comb.iloc[:train_border_index].Embarked.isnull().sum())
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    frequent_embarked = comb.iloc[:train_border_index].Embarked.mode()[0]
    comb.Embarked.fillna(frequent_embarked, inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(comb['Embarked'], prefix='Embarked')
    comb = pd.concat([comb, embarked_dummies], axis=1)
    comb.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return comb


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


def encode_sex(comb):
    # mapping string values to numerical one
    comb['Sex'] = comb['Sex'].map({'male': 1, 'female': 0})
    status('Sex')
    return comb


def encode_pclasses(comb):
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(comb['Pclass'], prefix="Pclass")

    # adding dummy variable
    comb = pd.concat([comb, pclass_dummies], axis=1)

    # removing "Pclass"
    comb.drop('Pclass', axis=1, inplace=True)

    status('Pclass')
    return comb


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


def split_combined_data(comb):
    en_train = comb[comb['is_test'] == 0]
    en_test = comb[comb['is_test'] == 1]
    return en_train, en_test


def extract_survived(data):
    return data['Survived']


def remove_passenger_id(data):
    return data.drop(['PassengerId'], 1)


def remove_is_test(data):
    return data.drop(['is_test'], 1)


def remove_age(data):
    return data.drop(['Age'], 1)


def remove_survived(data):
    return data.drop(['Survived'], 1)


# To avoid data leakage from the test set, we fill in missing ages in the train using the train set and we fill in ages
# in the test set using values calculated from the train set as well.
def predict_empty_ages(comb):
    data = comb[comb['is_test'] == 0]
    data = remove_is_test(data)
    x_train_age = data[data['Age'].notnull()]
    x_train_age = remove_age(x_train_age)
    x_train_age = remove_survived(x_train_age)
    y_train_age = data['Age']
    y_train_age = y_train_age[y_train_age.notnull()]

    mod = models.Sequential()
    mod.add(Dense(units=64, activation='relu', input_dim=x_train_age.shape[1]))
    mod.add(Dense(units=16, activation='relu'))
    mod.add(Dense(1, activation='linear'))

    mod.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    epo = 200
    mod.fit(x_train_age, y_train_age, epochs=epo, batch_size=512, verbose=2)

    train_predict_age = data[data['Age'].isnull()]
    train_predict_age = remove_age(train_predict_age)
    train_predict_age = remove_survived(train_predict_age)
    p_train = mod.predict(train_predict_age.values)
    # print(p_train)

    test_predict_age = comb[comb['is_test'] == 1]
    test_predict_age = remove_is_test(test_predict_age)
    test_predict_age = test_predict_age[test_predict_age['Age'].isnull()]
    test_predict_age = remove_age(test_predict_age)
    test_predict_age = remove_survived(test_predict_age)
    p_test = mod.predict(test_predict_age.values)

    p = np.rint(np.concatenate((p_train, p_test)))
    p = p.astype(int)
    p = p.flatten()
    # print(p)

    comb['Age'].loc[comb['Age'].isnull()] = p
    # show_data(comb, 'comb')
    return comb


def engineer_data():
    data = Data()
    data.raw_train = raw_train
    data.raw_train = raw_train
    data.train_border_index = train_border_index
    data.validation_border_index = validation_border_index

    combined = combine_data(raw_train, raw_test)
    combined = add_titles(combined)
    combined = fill_empty_ages(combined)
    combined = refine_names(combined)
    combined = encode_sex(combined)
    combined = encode_cabins(combined)
    combined = encode_pclasses(combined)
    combined = encode_tickets(combined)
    combined = fill_empty_fares(combined)
    combined = fill_empty_embarked(combined)
    combined = add_family_size(combined)
    combined = remove_passenger_id(combined)

    engineered_train, engineered_test = split_combined_data(combined)
    data.engineered_train = engineered_train
    data.engineered_test = engineered_train

    x_train_full = remove_survived(engineered_train.iloc[validation_border_index:train_border_index])
    x_train_full = remove_is_test(x_train_full)
    y_train_full = extract_survived(engineered_train.iloc[validation_border_index:train_border_index])

    x_train = x_train_full.iloc[validation_border_index:]
    y_train = y_train_full.iloc[validation_border_index:]

    x_val = x_train_full.iloc[:validation_border_index]
    y_val = y_train_full.iloc[:validation_border_index]

    x_test = remove_survived(engineered_test)
    x_test = remove_is_test(x_test)

    data.x_train_full = x_train_full
    data.y_train_full = y_train_full
    data.x_train = x_train
    data.y_train = y_train
    data.x_val = x_val
    data.y_val = y_val
    data.x_test = x_test

    return data
