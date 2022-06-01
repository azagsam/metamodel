import os
import string
from nltk.corpus import stopwords
import pandas as pd
import uuid
from tqdm import tqdm

from lemmagen3 import Lemmatizer


def filter_text(content):
    content_filtered = []
    for token in content.split():
        lemma = lem_sl.lemmatize(token)
        if lemma not in stopwords:
            content_filtered.append(lemma.lower())
    content_filtered = ' '.join(content_filtered)
    content_filtered = ''.join([i for i in content_filtered if not i.isdigit()])  # remove digits
    content_filtered = content_filtered.translate(str.maketrans('', '', string.punctuation))
    return content_filtered


def main():
    samples = []

    # get KAS
    for root, dirs, files in os.walk("/home/azagar/myfiles/kas_final/final/kas.corpus/kas.txt", topdown=False):
        for name in tqdm(files):
            if ".txt" in name:
                file_path = os.path.join(root, name)
                with open(file_path, 'r') as f:
                    content = f.read()
                    try:
                        content_filtered = filter_text(content)
                    except:
                        continue
                    samples.append({'text': content_filtered, 'id': str(uuid.uuid4()), 'source': 'kas'})

    # get STA+ASN
    root = '/home/azagar/myfiles/t5/data/asn-summary-plus-sta-lead'
    for mode in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
        file = os.path.join(root, mode)
        df = pd.read_json(file, lines=True)
        for content in tqdm(df['text'].to_list()):
            try:
                content_filtered = filter_text(content)
            except:
                continue
            samples.append({'text': content_filtered, 'id': str(uuid.uuid4()), 'source': 'news'})

    # get SURS
    df = pd.read_json('data/surs.jsonl', lines=True)
    for content in tqdm(df['KomentarSLO'].to_list()):
        try:
            content_filtered = filter_text(content)
        except:
            continue
        samples.append({'text': content_filtered, 'id': str(uuid.uuid4()), 'source': 'surs'})

    # export
    df = pd.DataFrame(samples)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_json('data/metamodel.jsonl', lines=True, orient='records', force_ascii=False)


if __name__ == '__main__':
    lem_sl = Lemmatizer('sl')
    stopwords = set(stopwords.words('slovene'))
    main()

