import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier

from python import feature_engineering as fe

pd.options.display.max_columns = 100

raw_train = pd.read_csv("../data/train.csv")
raw_test = pd.read_csv("../data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~82
def xgboost_data(data):
    return xgboost(data.x_train, data.y_train, data.x_val, data.y_val)


def xgboost(x_train, y_train, x_val, y_val):
    classifier = XGBClassifier()
    classifier.fit(x_train, y_train)
    return classifier.score(x_val, y_val)


# accuracy ~81
def xgboost_cross_validation(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    classifier = XGBClassifier()
    results = cross_val_score(classifier, data.x_train_full, data.y_train_full, cv=skf)
    return results.mean()


# accuracy ~83.7
def xgboost_cross_validation_best_params(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'n_estimators': [2, 5, 10, 25, 50, 100, 250, 500], 'max_depth': [1, 2, 3, 4, 5, 7, 10, 15, 20, 50],
                  'learning_rate': [0.001, 0.01, 0.1, 0.5, 1], 'booster': ['gbtree', 'gblinear', 'dart']}
    classifier = XGBClassifier()
    gcv = GridSearchCV(classifier, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(data.x_train_full, data.y_train_full)
    print(gcv.best_params_)
    return gcv.best_score_


# accuracy ~83.4
def gradient_boosting_cross_validation_best_params(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'n_estimators': [2, 5, 10, 25, 50, 100, 250, 500], 'max_depth': [1, 2, 3, 4, 5, 7, 10, 15, 20, 50],
                  'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]}
    classifier = GradientBoostingClassifier()
    gcv = GridSearchCV(classifier, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(data.x_train_full, data.y_train_full)
    print(gcv.best_params_)
    return gcv.best_score_


# accuracy ~83.7
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

    accuracy = gradient_boosting_cross_validation_best_params(data)
    print(accuracy)


# accuracy
if __name__ == "__main__":
    main()
