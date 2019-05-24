import pandas as pd
import data_analysis as da
import feature_engineering as fe
import model_analysis as ma


import tensorflow
from keras import models
from keras.legacy import layers
from keras.layers import Dense, Dropout

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


if __name__ == "__main__":
    main()
