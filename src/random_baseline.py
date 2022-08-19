import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def mean_predict(data, embeddings):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p['validation_split'], random_state=p['seed'])

    # random predict
    y_pred = pd.concat([y_train.mean(axis=0).to_frame().T]*len(y_test)).reset_index(drop=True)
    results = {'mse_test': mean_squared_error(y_test, y_pred)}

    with open('/home/azagar/myfiles/metamodel/baseline-metrics.json', 'w') as outfile:
        json.dump(results, outfile)

if __name__ == '__main__':
    data = sys.argv[1]
    embeddings = sys.argv[2]

    mean_predict(data, embeddings)
