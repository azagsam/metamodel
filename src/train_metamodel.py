import json
import os.path

import numpy as np
import pandas as pd
import yaml
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
from lemmagen3 import Lemmatizer
from nltk.corpus import stopwords
import string
import tensorflow as tf
import sys


def prepare_embeddings(data, stopwords, d2v_model, save_path):
    def filter_text(content):
        content_filtered = []
        for token in content.split():
            lemma = lem_sl.lemmatize(token)
            if lemma not in stopwords:
                content_filtered.append(lemma.lower())
        content_filtered = ' '.join(content_filtered)
        content_filtered = ''.join([i for i in content_filtered if not i.isdigit()])  # remove digits
        content_filtered = content_filtered.translate(str.maketrans('', '', string.punctuation))
        return content_filtered


    # get and prepare data
    print('Retrieving data ... ')
    df = pd.read_json(data, lines=True)
    print('Retrieved.')

    # load doc2vec model & load external tools
    lem_sl = Lemmatizer('sl')
    stopwords = set(stopwords.words('slovene'))
    model = Doc2Vec.load(d2v_model)
    doc_vectors = []
    for text in tqdm(df['text']):
        doc_vec = model.infer_vector(filter_text(text).split())
        doc_vectors.append(doc_vec)
    doc_vectors = np.array(doc_vectors)

    # save
    np.save(save_path, doc_vectors)  # .npy extension is added if not given


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
    d2v_model = sys.argv[3]
    model_save_path = sys.argv[4]

    # prepare embeddings if file does not exist
    if not os.path.isfile(embeddings):
        prepare_embeddings(data, stopwords, d2v_model=d2v_model, save_path=embeddings)

    train(data, embeddings, model_save_path)