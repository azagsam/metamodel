import os

from gensim.models import Doc2Vec
from lemmagen3 import Lemmatizer
from tensorflow import keras
from nltk.corpus import stopwords
import string
import pandas as pd
from tqdm import tqdm
from collections import Counter


def filter_text(content, lem_sl, stopwords):
    content_filtered = []
    for token in content.split():
        lemma = lem_sl.lemmatize(token)
        if lemma not in stopwords:
            content_filtered.append(lemma.lower())
    content_filtered = ' '.join(content_filtered)
    content_filtered = ''.join([i for i in content_filtered if not i.isdigit()])  # remove digits
    content_filtered = content_filtered.translate(str.maketrans('', '', string.punctuation))
    return content_filtered


def get_recommended_model(d2v_model, metamodel, text):
    # preprocess and score
    preprocessed_text = filter_text(text, lem_sl, stopwords).split()
    doc_vector = d2v_model.infer_vector(preprocessed_text)
    scores = metamodel.predict(doc_vector.reshape(1, -1))

    # Scores in group of four: verify the correct order: t5-article, graph-based, hybrid-long, sumbasic
    t5_article = scores[:, 0:4]
    graph_based = scores[:, 4:8]
    hybrid_long = scores[:, 8:12]
    sumbasic = scores[:, 12:]

    averages = [
        ('t5-article', t5_article.mean()),
        ('graph-based', graph_based.mean()),
        ('hybrid-long', hybrid_long.mean()),
        ('sumbasic', sumbasic.mean())
    ]
    averages.sort(key=lambda x: x[1], reverse=True)

    return averages[0][0]


if __name__ == '__main__':
    fname = "model/doc2vec/model"
    d2v_model = Doc2Vec.load(fname)
    lem_sl = Lemmatizer('sl')
    stopwords = set(stopwords.words('slovene'))

    # load metamodel
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    metamodel = keras.models.load_model('model/metamodel/model.h5')

    # import data
    df = pd.read_json('data/final/test.jsonl', lines=True)
    sample = df.sample(n=1000)

    recommended = []
    with open('data/manual_analysis.txt', 'w') as f:
        for idx, row in tqdm(sample.iterrows()):
            text = row['text']
            d2v_recommendation = get_recommended_model(d2v_model, metamodel, text)
            recommended.append(d2v_recommendation)

            f.write('ID: ')
            f.write(row['id'])
            f.write('\n\n')

            f.write(row['text'])
            f.write('\n\n')

            f.write('t5-article: ')
            f.write(row['t5-article'])
            f.write('\n')
            f.write(str(row['t5-article-score']))
            f.write('\n\n')

            f.write('graph-based: ')
            f.write(row['graph-based'])
            f.write('\n')
            f.write(str(row['graph-based-score']))
            f.write('\n\n')

            f.write('hybrid-long: ')
            f.write(row['hybrid-long'])
            f.write('\n')
            f.write(str(row['hybrid-long-score']))
            f.write('\n\n')

            f.write('sumbasic: ')
            f.write(row['sumbasic'])
            f.write('\n')
            f.write(str(row['sumbasic-score']))
            f.write('\n\n')

            f.write(f'd2v: {d2v_recommendation}')
            f.write('\n')
            f.write('human:')
            f.write(5*'\n')

    count = Counter(recommended)
    print('Count: ', count)
