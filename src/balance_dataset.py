import pandas as pd

df = pd.read_json('data/metamodel-training-with-scores-large.jsonl', lines=True)
df = df[df['source'] != 'surs']
g = df.groupby('source')
balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
balanced.to_json('data/metamodel-training-with-scores-balanced.jsonl', lines=True, orient='records', force_ascii=False)