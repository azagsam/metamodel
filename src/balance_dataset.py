import pandas as pd

df = pd.read_json('data/metamodel-training-with-scores-large.jsonl', lines=True)

print(df.head())