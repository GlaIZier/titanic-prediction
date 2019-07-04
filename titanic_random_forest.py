import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
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


# accuracy ~82
def random_forest_cross_validation(data, splits=5):

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
    results = cross_val_score(rfc, data.x_train_full, data.y_train_full, cv=skf)
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

    accuracy = random_forest_cross_validation(data)
    print(accuracy)


# accuracy ~83
if __name__ == "__main__":
    main()
