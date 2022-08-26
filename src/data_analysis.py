import pandas as pd

df = pd.read_json('data/metamodel-training-with-scores-large.jsonl', lines=True)

df['source'] = df['source'].apply(lambda x: 'long' if x == 'kas' else 'short')

print(df.head())
grouped_source = df.groupby('source').mean().reset_index()

models = ['t5-article', 'graph-based', 'hybrid-long', 'sumbasic']
for model in models:
    res = grouped_source.filter(regex=model).mean(axis=1).round(2)
    print(model)
    print(res)