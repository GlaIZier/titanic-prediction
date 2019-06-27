import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


def display_decision_tree(decision_tree_model, x):
    with open("dtree.dot", 'w') as dotfile:
        tree.export_graphviz(decision_tree_model, out_file=dotfile, feature_names=x.columns, class_names=True)


# accuracy ~81
def decision_tree_data(data):
    accuracy = decision_tree(data.x_train, data.y_train, data.x_val, data.y_val)
    return accuracy


def decision_tree(x_train, y_train, x_val, y_val):
    decision_tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
    decision_tree_model.fit(x_train, y_train)
    # display_decision_tree(decision_tree_model, x_train)
    return decision_tree_model.score(x_val, y_val)


# accuracy ~82
def decision_tree_cross_validation(data, splits=5):

    kf = KFold(n_splits=splits)
    accuracy = 0
    for train_indexes, val_indexes in kf.split(data.x_train_full):
        x_train = data.x_train_full.iloc[train_indexes]
        y_train = data.y_train_full.iloc[train_indexes]
        x_val = data.x_train_full.iloc[val_indexes]
        y_val = data.y_train_full.iloc[val_indexes]
        accuracy += decision_tree(x_train, y_train, x_val, y_val)

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

    accuracy = decision_tree_cross_validation(data)
    print(accuracy)


# accuracy ~80
if __name__ == "__main__":
    main()
