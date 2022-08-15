import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


# get and prepare data
print('Retrieving data ... ')
df = pd.read_json('data/metamodel.jsonl', lines=True)

texts = [text.split() for text in df['text']]
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

# train model
print('Model training ...')
epoch_logger = EpochLogger()
model = Doc2Vec(documents,
                vector_size=128,
                window=5,
                min_count=1,
                workers=16,
                callbacks=[epoch_logger],
                max_vocab_size=100000)

# save model
fname = "model/model-small/metamodel"
model.save(fname)

# load model
model = Doc2Vec.load(fname)