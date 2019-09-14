import pandas as pd
from python import feature_engineering as fe, model_analysis as ma

from keras import models
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold


pd.options.display.max_columns = 100

raw_train = pd.read_csv("../data/train.csv")
raw_test = pd.read_csv("../data/test.csv")
train_border_index = 891
validation_border_index = 265


def nn_keras(data):
    # 3 Model development and prediction
    model = models.Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=data.x_train.shape[1]))
    model.add(Dropout(0.25))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    epochs = 1000
    history = model.fit(data.x_train, data.y_train, epochs=epochs, batch_size=512, verbose=2,
                        validation_data=[data.x_val, data.y_val])

    # 4. Model analysis
    ma.plot_train_val_loss(history, epochs)
    ma.plot_train_val_acc(history, epochs)


def nn_keras_cross_validation(data):
    def create_nn():
        model = models.Sequential()
        model.add(Dense(1024, input_dim=data.x_train_full.shape[1], activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=create_nn, epochs=20, batch_size=10, verbose=1)
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    results = cross_val_score(estimator, data.x_train_full, data.y_train_full, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


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

    # 3 Model development and prediction
    nn_keras_cross_validation(data)


# ~ 80%
if __name__ == "__main__":
    main()
