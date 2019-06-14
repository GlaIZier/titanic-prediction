import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~81
def linear_regression(data):
    linear_regression = LinearRegression()
    linear_regression.fit(data.x_train, data.y_train)
    y_pred = linear_regression.predict(data.x_val)
    threshold = 1 - data.y_train.mean()
    y_pred = (y_pred > threshold).astype(float)
    accuracy = accuracy_score(data.y_val, y_pred)
    print(accuracy)


# # accuracy ~81
# def linear_regression_cross_validation(data):
#     accuracy = cross_val_score(LinearRegression(), data.x_train_full, data.y_train_full, cv=5)\
#         .mean()
#     print(accuracy)


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

    linear_regression(data)


# accuracy ~81
if __name__ == "__main__":
    main()
