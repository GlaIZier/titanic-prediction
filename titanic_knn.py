import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier

import data_analysis as da
import feature_engineering as fe

from keras import models
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~70
def knn(data):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data.x_train, data.y_train)
    accuracy = knn.score(data.x_val, data.y_val)
    print(accuracy)

# accuracy ~81
def logistic_regression_cross_validation(data):
    model = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(data.x_train, data.y_train)
    accuracy = model.score(data.x_val, data.y_val)
    print(accuracy)


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

    knn(data)


# accuracy ~81
if __name__ == "__main__":
    main()
