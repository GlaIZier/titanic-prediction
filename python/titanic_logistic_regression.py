import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from python import feature_engineering as fe

pd.options.display.max_columns = 100

raw_train = pd.read_csv("../data/train.csv")
raw_test = pd.read_csv("../data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~81
def logistic_regression(data):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(data.x_train, data.y_train)
    accuracy = logistic_regression.score(data.x_val, data.y_val)
    print(accuracy)


# accuracy ~81
def logistic_regression_cross_validation(data):
    model = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(data.x_train, data.y_train)
    accuracy = model.score(data.x_val, data.y_val)
    print(accuracy)


# accuracy ~68
def logistic_regression_sgd_cross_validation(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    results = cross_val_score(SGDClassifier(random_state=42),
                              data.x_train_full, data.y_train_full, cv=skf)
    return results.mean()


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

    logistic_regression_cross_validation(data)
    print(logistic_regression_sgd_cross_validation(data))


# accuracy ~81
if __name__ == "__main__":
    main()
