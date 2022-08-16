import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def train(data, embeddings, model_save_path):
    # get and prepare data
    print('Retrieving data ... ')
    df = pd.read_json(data, lines=True)
    doc_vectors = np.load(embeddings)
    print('Retrieved.')

    X = np.array(doc_vectors)
    y = df.filter(regex='rouge')

    # params
    p = yaml.safe_load(open('/home/azagar/myfiles/metamodel/params.yaml'))['train-metamodel']

    # split data into train/val
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=p['seed'])

    # build model
    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # define callback
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # train
    history = model.fit(X_train, y_train,
                        batch_size=8,
                        epochs=100,
                        verbose=1,
                        validation_split=.2,
                        callbacks=[callback])

    # save or load model
    model.save(model_save_path)

    # history
    metrics = {k: v[-1] for k, v in history.history.items()}
    with open('/home/azagar/myfiles/metamodel/metamodel-metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)


if __name__ == '__main__':
    data = sys.argv[1]
    embeddings = sys.argv[2]
    model_save_path = sys.argv[3]

    train(data, embeddings, model_save_path)
