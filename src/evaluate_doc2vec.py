import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

# get and prepare data
print('Retrieving data ... ')
df = pd.read_json('/home/azagar/myfiles/metamodel/data/doc2vec-training.jsonl', lines=True)

# documents = df['text'].to_list()[:10000]
# del df

# load model
fname = "/home/azagar/myfiles/metamodel/model/model-large/metamodel"
model = Doc2Vec.load(fname)

vectors = np.array([model.infer_vector(df['text'][n].split()) for n in list(range(50000))])
cos_vectors = cosine_similarity(vectors, vectors)
for idx, cos_vec in enumerate(cos_vectors):
    print('\n' * 3)
    indices = np.argsort(cos_vec)[-5:]
    print('SOURCE', df['text'][idx])
    for idx_target in indices:
        print('\n')
        print(df['text'][idx_target])
    if idx == 5:
        break


model.wv.most_similar('računalnik', topn=10)

# # cluster and write to disk
# eps = 0.3
# min_samples = 5
# print('Start DBSCAN ...')
# start_time = time.time()
# clusters = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples).fit_predict(doc_vectors)
# print("DBSCAN finished in --- %s seconds ---" % (time.time() - start_time))
#
# grouped = {}
# for _i, cl in enumerate(clusters):
#     if cl != -1:
#         current_cl = grouped.get(cl, [])
#         current_cl.append(documents[_i])
#         grouped[cl] = current_cl
#
# with open(f"metamodel_results-eps_{eps}-minsamples_{min_samples}.txt", "w", encoding="utf-8") as f:
#     for cl, items in grouped.items():
#         print(f"Cluster#{cl}:", file=f)
#         for ex in items:
#             print(f"\t- {ex}", file=f)
#         print("", file=f)