import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~79
def extra_trees_data(data):
    return mlp(data.x_train, data.y_train, data.x_val, data.y_val)


def mlp(x_train, y_train, x_val, y_val):
    classifier = MLPClassifier(random_state=42)
    classifier.fit(x_train, y_train)
    return classifier.score(x_val, y_val)


# accuracy ~82.6
def mlp_cross_validation(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    classifier = MLPClassifier(random_state=42)
    results = cross_val_score(classifier, data.x_train_full, data.y_train_full, cv=skf)
    return results.mean()


# accuracy ~83.4
def mlp_cross_validation_best_params(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'hidden_layer_sizes': [(512, ), (128, ), (16, ), (512, 64, ), (128, 16), (16, 4)],
                  'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 'max_iter': [50, 100, 200, 400, 750, 1000],
                  'early_stopping': [False, True]}
    classifier = MLPClassifier(random_state=42)
    gcv = GridSearchCV(classifier, parameters, n_jobs=-1, cv=skf, verbose=3)
    gcv.fit(data.x_train_full, data.y_train_full)
    print(gcv.best_params_)
    return gcv.best_score_


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

    accuracy = mlp_cross_validation_best_params(data)
    print(accuracy)


# accuracy ~83
if __name__ == "__main__":
    main()
