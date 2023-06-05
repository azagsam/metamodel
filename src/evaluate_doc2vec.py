import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_metric


def compute_metrics(source, target):
    # Rouge expects a newline after each sentence
    source = source.split()
    target = target.split()

    result = metric.compute(predictions=[target], references=[source], use_stemmer=True)

    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return {k: round(v, 4) for k, v in result.items()}



# get and prepare data
print('Retrieving data ... ')
df = pd.read_json('/home/azagar/myfiles/metamodel/data/doc2vec-training.jsonl', lines=True)
print('Retrieved.')

# load metric
metric = load_metric("rouge")

# load model
fname = "/home/azagar/myfiles/metamodel/model/doc2vec/model"
model = Doc2Vec.load(fname)

vectors = np.array([model.infer_vector(df['text'][n].split()) for n in tqdm(list(range(50000)))])
cos_vectors = cosine_similarity(vectors, vectors)
for idx, cos_vec in enumerate(cos_vectors):
    print(idx)
    print('\n' * 3)
    indices = np.argsort(cos_vec)[-4:-1]
    print('SOURCE', df['text'][idx][:300])
    for idx_target in indices:
        print('\n')
        print(df['text'][idx_target][:300])
        print(compute_metrics(df['text'][idx], df['text'][idx_target]))
    if idx == 30:
        break

# evaluate word embeddings
print(model.wv.most_similar('računalnik', topn=10))
