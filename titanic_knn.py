import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~70
def knn_data(data):
    accuracy = knn(data.x_train, data.y_train, data.x_val, data.y_val)
    print(accuracy)


def knn(x_train, y_train, x_val, y_val):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    return knn.score(x_val, y_val)


# accuracy ~73
def knn_cross_validation(data, splits=5):

    kf = KFold(n_splits=splits)
    accuracy = 0
    for train_indexes, val_indexes in kf.split(data.x_train_full):
        x_train = data.x_train_full.iloc[train_indexes]
        y_train = data.y_train_full.iloc[train_indexes]
        x_val = data.x_train_full.iloc[val_indexes]
        y_val = data.y_train_full.iloc[val_indexes]
        accuracy += knn(x_train, y_train, x_val, y_val)

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

    accuracy = knn_cross_validation(data)
    print(accuracy)


# accuracy ~73
if __name__ == "__main__":
    main()
