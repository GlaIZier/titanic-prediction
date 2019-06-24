import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB

import data_analysis as da
import feature_engineering as fe


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


# accuracy ~81
def naive_bayes(data):
    gaussian_naive_bayes = GaussianNB()
    gaussian_naive_bayes.fit(data.x_train, data.y_train)
    accuracy = gaussian_naive_bayes.score(data.x_val, data.y_val)
    return accuracy


# accuracy ~81
def naive_bayes_cross_validation(data):
    model = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(data.x_train, data.y_train)
    return model.score(data.x_val, data.y_val)


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

    accuracy = naive_bayes(data)
    print(accuracy)


# accuracy ~81
if __name__ == "__main__":
    main()
