import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


def linear_regression(x_train, y_train, x_val, y_val):
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    y_pred = linear_regression.predict(x_val)
    # threshold = 1 - y_train.mean()
    threshold = 0.5
    y_pred = (y_pred > threshold).astype(float)
    return accuracy_score(y_val, y_pred)


# accuracy ~81
def linear_regression_data(data):
    accuracy = linear_regression(data.x_train, data.y_train, data.x_val, data.y_val)
    print(accuracy)


# accuracy ~83
def linear_regression_cross_validation(data, splits=5):

    kf = KFold(n_splits=splits)
    accuracy = 0
    for train_indexes, val_indexes in kf.split(data.x_train_full):
        x_train = data.x_train_full.iloc[train_indexes]
        y_train = data.y_train_full.iloc[train_indexes]
        x_val = data.x_train_full.iloc[val_indexes]
        y_val = data.y_train_full.iloc[val_indexes]
        accuracy += linear_regression(x_train, y_train, x_val, y_val)

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

    accuracy = linear_regression_cross_validation(data)
    print(accuracy)


# accuracy ~83
if __name__ == "__main__":
    main()
