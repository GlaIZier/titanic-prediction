import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC

from python import feature_engineering as fe

pd.options.display.max_columns = 100

raw_train = pd.read_csv("../data/train.csv")
raw_test = pd.read_csv("../data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~82 (linear),
def svm_data(data, kernel='linear'):
    accuracy = svm(data.x_train, data.y_train, data.x_val, data.y_val, kernel)
    return accuracy


def svm(x_train, y_train, x_val, y_val, kernel='linear'):
    gaussian_naive_bayes = SVC(kernel=kernel)
    gaussian_naive_bayes.fit(x_train, y_train)
    return gaussian_naive_bayes.score(x_val, y_val)


# accuracy ~83 (linear)
def svm_cross_validation_manual(data, splits=5, kernel='linear'):
    kf = KFold(n_splits=splits)
    accuracy = 0
    for train_indexes, val_indexes in kf.split(data.x_train_full):
        x_train = data.x_train_full.iloc[train_indexes]
        y_train = data.y_train_full.iloc[train_indexes]
        x_val = data.x_train_full.iloc[val_indexes]
        y_val = data.y_train_full.iloc[val_indexes]
        accuracy += svm(x_train, y_train, x_val, y_val, kernel)

    return accuracy / splits


# accuracy ~83 (linear); 79 (poly); 73 (rbf); 61 (sigmoid)
def svm_cross_validation(data, splits=5, kernel='linear'):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    results = cross_val_score(SVC(kernel=kernel), data.x_train_full, data.y_train_full, cv=skf)
    return results.mean()


# accuracy ~81 (rbf, C = 10, gamma = 0.001)
def svm_cross_validation_best_params_rbf_sigmoid(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'kernel': ['rbf', 'sigmoid'], 'C': [0.01, 0.1, 1, 10, 50],
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    classifier = SVC(random_state=42)
    gcv = GridSearchCV(classifier, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(data.x_train_full, data.y_train_full)
    print(gcv.best_params_)
    return gcv.best_score_


# very slow!
def svm_cross_validation_best_params_poly(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'degree': [1, 2, 4, 6, 8, 10], 'C': [0.01, 0.1, 1, 10, 50],
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    classifier = SVC(random_state=42, kernel='poly')
    gcv = GridSearchCV(classifier, parameters, n_jobs=-1, cv=skf, verbose=3)
    gcv.fit(data.x_train_full, data.y_train_full)
    print(gcv.best_params_)
    return gcv.best_score_


# accuracy 81%
def svm_cross_validation_poly_with_best_params(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    results = cross_val_score(SVC(kernel='poly', C=0.1, degree=2, gamma=0.01), data.x_train_full, data.y_train_full,
                              cv=skf)
    return results.mean()


# accuracy 83% C=0.1, gamma = 0.0001
def svm_cross_validation_best_params_linear(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'C': [0.01, 0.1, 1, 10, 50],
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    classifier = SVC(random_state=42, kernel='linear')
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

    accuracy = svm_cross_validation_best_params_linear(data)
    print(accuracy)


if __name__ == "__main__":
    main()
