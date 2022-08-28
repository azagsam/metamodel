import os

import numpy as np
import pandas as pd

# create splits
df = pd.read_json('data/metamodel-training-with-scores-large.jsonl', lines=True)
val, test = .90, .95
target_dir = 'data/final'
train, val, test = np.split(df.sample(frac=1, random_state=42), [int(val * len(df)), int(test * len(df))])

os.makedirs(target_dir, exist_ok=True)
train.to_json(os.path.join(target_dir, 'train.jsonl'), lines=True, force_ascii=False, orient='records')
val.to_json(os.path.join(target_dir, 'val.jsonl'), lines=True, force_ascii=False, orient='records')
test.to_json(os.path.join(target_dir, 'test.jsonl'), lines=True, force_ascii=False, orient='records')

# create balanced
df = pd.read_json(f'{target_dir}/train.jsonl', lines=True)
df = df[df['source'] != 'surs']
g = df.groupby('source')
balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
balanced.to_json(f'{target_dir}/train-balanced.jsonl', lines=True, orient='records', force_ascii=False)