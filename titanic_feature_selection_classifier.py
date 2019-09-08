import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

import feature_engineering as fe

pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~83
def extra_trees_data(data):
    return extra_trees(data.x_train, data.y_train, data.x_val, data.y_val)


def extra_trees(x_train, y_train, x_val, y_val):
    classifier = ExtraTreesClassifier(random_state=42)
    classifier.fit(x_train, y_train)
    return classifier.score(x_val, y_val)


# accuracy ~82
def extra_trees_cross_validation(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    classifier = ExtraTreesClassifier(random_state=42)
    results = cross_val_score(classifier, data.x_train_full, data.y_train_full, cv=skf)
    return results.mean()


# accuracy ~84.6 : 5, 10, 8
def extra_trees_cross_validation_best_params(data, splits=5):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'n_estimators': [2, 5, 10, 25, 50, 100, 250, 500],
                  'max_depth': [1, 2, 5, 7, 10, 15, 20, 50, 100, None], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 10,
                                                                                              15]}
    classifier = ExtraTreesClassifier(random_state=42)
    gcv = GridSearchCV(classifier, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(data.x_train_full, data.y_train_full)
    print("Best params: " + gcv.best_params_)
    return gcv.best_score_


# accuracy ~82.6
def extra_trees_feature_selection(data, splits=5):
    # print(data.x_train_full.shape)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    classifier = ExtraTreesClassifier(random_state=42, n_estimators=5, max_depth=10, min_samples_split=8)
    classifier = classifier.fit(data.x_train_full, data.y_train_full)
    print("Importance of features: " + classifier.feature_importances_)

    model = SelectFromModel(classifier, prefit=True)
    selected_x = model.transform(data.x_train_full)
    print("Selected shape: " + selected_x.shape)

    results = cross_val_score(classifier, selected_x, data.y_train_full, cv=skf)
    return results.mean()


# accuracy ~84.2
def extra_recursive_feature_selection(data, splits=5):
    # print(data.x_train_full.shape)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    classifier = ExtraTreesClassifier(random_state=42, n_estimators=5, max_depth=10, min_samples_split=8)

    rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(5),
                  scoring='accuracy')
    rfecv.fit(data.x_train, data.y_train)

    print('Optimal number of features: ' + str(rfecv.n_features_))

    selected_x_train = rfecv.transform(data.x_train)
    selected_x_val = rfecv.transform(data.x_val)
    selected_x_train_full = rfecv.transform(data.x_train_full)

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=17)
    parameters = {'n_estimators': [2, 5, 10, 25, 50, 100, 250, 500],
                  'max_depth': [1, 2, 5, 7, 10, 15, 20, 50, 100, None], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 10,
                                                                                              15]}
    classifier = ExtraTreesClassifier(random_state=42)
    gcv = GridSearchCV(classifier, parameters, n_jobs=-1, cv=skf, verbose=1)
    gcv.fit(selected_x_train_full, data.y_train_full)
    print("Best params: " + repr(gcv.best_params_))
    return gcv.best_score_

    # model = SelectFromModel(classifier, prefit=True)
    # selected_x = model.transform(data.x_train_full)
    # print(selected_x.shape)
    #
    # results = cross_val_score(classifier, selected_x, data.y_train_full, cv=skf)
    # return results.mean()


def make_pipeline_classifier(data):
    pass


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

    accuracy = extra_recursive_feature_selection(data)
    print(accuracy)


# accuracy ~84.6
if __name__ == "__main__":
    main()
