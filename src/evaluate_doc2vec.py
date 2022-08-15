import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import time
import numpy as np
from collections import Counter

# get and prepare data
print('Retrieving data ... ')
df = pd.read_json('data/metamodel.jsonl', lines=True)

# documents = df['text'].to_list()[:10000]
# del df

# load model
fname = "model/model-large/metamodel"
model = Doc2Vec.load(fname)
# doc_vectors = model.dv.vectors[:10000]

# infer vector for a new text
for n in list(range(100)):
    print('\n'*3)
    new_vector = model.infer_vector(df['text'][n].split())
    print(df['text'][n])
    for idx, score in model.dv.most_similar([new_vector], topn=5):
        print('\n')
        print(idx, score, df['text'][idx])


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