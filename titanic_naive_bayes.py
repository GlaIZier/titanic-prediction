import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~50
def naive_bayes_data(data, classificator=GaussianNB()):
    accuracy = naive_bayes(data.x_train, data.y_train, data.x_val, data.y_val, classificator)
    return accuracy


def naive_bayes(x_train, y_train, x_val, y_val, classificator=GaussianNB()):
    gaussian_naive_bayes = classificator
    gaussian_naive_bayes.fit(x_train, y_train)
    return gaussian_naive_bayes.score(x_val, y_val)


# accuracy ~52
def naive_bayes_cross_validation(data, splits=5, classificator=GaussianNB()):

    kf = KFold(n_splits=splits)
    accuracy = 0
    for train_indexes, val_indexes in kf.split(data.x_train_full):
        x_train = data.x_train_full.iloc[train_indexes]
        y_train = data.y_train_full.iloc[train_indexes]
        x_val = data.x_train_full.iloc[val_indexes]
        y_val = data.y_train_full.iloc[val_indexes]
        accuracy += naive_bayes(x_train, y_train, x_val, y_val, classificator)

    return accuracy / splits


def main():
    # 1. Data analysis
    # da.show_data(raw_train, 'raw train set:')
    # da.show_data(raw_test, 'raw test set: ')
    # da.analyze_training_data(raw_train)

    # 2. Feature engineering
    fe.raw_train = raw_train
    fe.raw_test = raw_test
    fe.train_border_index = train_border_index
    fe.validation_border_index = validation_border_index
    data = fe.engineer_data()

    accuracy = naive_bayes_cross_validation(data)
    print(accuracy)


# accuracy ~52
if __name__ == "__main__":
    main()
