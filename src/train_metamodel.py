import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def train(data, embeddings, model_save_path):
    # get and prepare data
    print('Retrieving data ... ')
    df = pd.read_json(data, lines=True)
    doc_vectors = np.load(embeddings)
    print('Retrieved.')

    X = np.array(doc_vectors)
    y = df.filter(regex='rouge')

    # add length info
    print('Adding length info ...')
    lengths = np.array([len(t) for t in df['text']]).reshape(-1, 1)
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(lengths)
    scaled = scaler.transform(lengths)

    X_length = []
    for vec, num in zip(X, scaled):
        vec = np.append(vec, np.float32(num))
        X_length.append(vec)
    X = np.array(X_length)

    # params
    p = yaml.safe_load(open('params.yaml'))['metamodel']

    # split data into train/val
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p['validation_split'], random_state=p['seed'])

    # build model
    model = Sequential()
    model.add(Dense(p['hidden_layer_size'], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(p['hidden_layer_size'], activation='relu'))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    print(p['hidden_layer_size'])
    # define callback
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=p['patience'])

    # train
    history = model.fit(X_train, y_train,
                        batch_size=8,
                        epochs=100,
                        verbose=1,
                        validation_split=p['validation_split'],
                        callbacks=[callback])

    # save or load model
    model.save(model_save_path)

    # predict on test set
    y_pred = model.predict(X_test)
    results = {'mse_test': mean_squared_error(y_test, y_pred)}

    # history
    metrics = {k: v[-1] for k, v in history.history.items()}
    metrics.update(results)
    with open('advanced-metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)


if __name__ == '__main__':
    data = sys.argv[1]
    embeddings = sys.argv[2]
    model_save_path = sys.argv[3]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.makedirs('model/metamodel', exist_ok=True)

    train(data, embeddings, model_save_path)
