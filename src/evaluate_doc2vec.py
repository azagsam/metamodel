import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# get and prepare data
print('Retrieving data ... ')
df = pd.read_json('/home/azagar/myfiles/metamodel/data/doc2vec-training.jsonl', lines=True)
print('Retrieved.')

# load model
fname = "/home/azagar/myfiles/metamodel/model/doc2vec/model"
model = Doc2Vec.load(fname)

vectors = np.array([model.infer_vector(df['text'][n].split()) for n in tqdm(list(range(50000)))])
cos_vectors = cosine_similarity(vectors, vectors)
for idx, cos_vec in enumerate(cos_vectors):
    print('\n' * 3)
    indices = np.argsort(cos_vec)[-3:-1]
    print('SOURCE', df['text'][idx])
    for idx_target in indices:
        print('\n')
        print(df['text'][idx_target])
    if idx == 25:
        break

# evaluate word embeddings
print(model.wv.most_similar('računalnik', topn=10))
