import pandas as pd
import data_analysis as da
import feature_engineering as fe
import model_analysis as ma


import tensorflow
from keras import models
from keras.legacy import layers
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold


pd.options.display.max_columns = 100

raw_train = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")
train_border_index = 891
validation_border_index = 265


def main():
    # 1. Data analysis
    da.show_data(raw_train, 'raw train set:')
    da.show_data(raw_test, 'raw test set: ')
    da.analyze_training_data(raw_train)

    # 2. Feature engineering
    fe.raw_train = raw_train
    fe.raw_test = raw_test
    fe.train_border_index = train_border_index
    fe.validation_border_index = validation_border_index
    data = fe.engineer_data()

    # 3 Model development and prediction
    def create_baseline():
        model = models.Sequential()
        model.add(Dense(1024, input_dim=data.x_train_full.shape[1], activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=create_baseline, epochs=20, batch_size=10, verbose=1)
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    results = cross_val_score(estimator, data.x_train_full, data.y_train_full, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


if __name__ == "__main__":
    main()
