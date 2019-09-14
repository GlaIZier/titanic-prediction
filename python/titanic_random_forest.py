import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

from python import feature_engineering as fe

pd.options.display.max_columns = 100

raw_train = pd.read_csv("../data/train.csv")
raw_test = pd.read_csv("../data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~85
def random_forest_data(data):
    accuracy = random_forest(data.x_train, data.y_train, data.x_val, data.y_val)
    print(accuracy)


def random_forest(x_train, y_train, x_val, y_val):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    return random_forest.score(x_val, y_val)


# accuracy ~83
def random_forest_cross_validation_manual(data, splits=5):

    kf = KFold(n_splits=splits)
    accuracy = 0
    for train_indexes, val_indexes in kf.split(data.x_train_full):
        x_train = data.x_train_full.iloc[train_indexes]
        y_train = data.y_train_full.iloc[train_indexes]
        x_val = data.x_train_full.iloc[val_indexes]
        y_val = data.y_train_full.iloc[val_indexes]
        accuracy += random_forest(x_train, y_train, x_val, y_val)

    return accuracy / splits


# accuracy ~82- ~84 (with best params)
def random_forest_cross_validation(data, splits=5, model_params=None):

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True) if model_params is None else \
        RandomForestClassifier(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'],
                               max_features=model_params['max_features'],
                               min_samples_leaf=model_params['min_samples_leaf'],
                               random_state=42, n_jobs=-1, oob_score=True)
    results = cross_val_score(rfc, data.x_train_full, data.y_train_full, cv=skf)
    return results.mean()


def choose_best_params(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'n_estimators': [25, 50, 75, 100, 150, 200, 500], 'max_features': [4, 7, 10, 13],
                  'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5, 10, 15, 20]}
    rfc = RandomForestClassifier(random_state=42,
                                 n_jobs=-1, oob_score=True)
    gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(data.x_train_full, data.y_train_full)
    return gcv.best_params_


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

    accuracy = random_forest_cross_validation(data, model_params=choose_best_params(data))
    print(accuracy)


# accuracy ~83
if __name__ == "__main__":
    main()
