'''
This file pre-processes the UD datasets.
'''
import argparse
import json
import random
import re
import xml.etree.ElementTree as ET

import transformers as hf

SENT_RE = re.compile(r'(?:.+\n)+.+(?=[\r\n]{2,}|$)')
FEAT_RE = re.compile(r'\t+|(?<=\w)\|(?=\w)')

EXCL_POS = ('INTJ', 'PUNCT', 'SYM', 'X')
EXCL_RE = re.compile(r'Abbr|Foreign|HebSource|NumForm|Punct|Style|Typo|Xtra')
VALS_RE = re.compile(r'(?<=[a-z])\B(?=[A-Z])|,')


def clean_corpus(in_corpus_fn, min_len=3, max_len=512):
    ''' '''
    # delete language-specific comment lines
    with open(in_corpus_fn, 'rb') as f:
        text = f.read().decode('utf-8')
        text = re.sub(r'(?:\n|^)# (?!sent_id =|text =).+', '', text)

    # instantiate BERT tokenizer
    tokenizer = hf.BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased',
        do_lower_case=False,
        do_basic_tokenize=False,
        )

    # make room for [CLS] and [SEP] tokens
    max_len -= 2

    out = []

    for sent in SENT_RE.findall(text):
        lines = (FEAT_RE.split(i) for i in sent.split('\n')[2:])
        tokens = []  # UD tokens

        try:
            while True:
                line = next(lines)
                idx, tok = line[0], line[1]
                tokens.append(tok)

                if '-' in idx:  # multiword token
                    i, j = (int(_) for _ in idx.split('-'))
                    for _ in range(i, j + 1):
                        next(lines)

        except StopIteration:
            pieces = tokenizer.tokenize(' '.join(tokens))  # BERT wordpieces

            # exclude the sentence if its wordpiece sequence length is less
            # than `min_len` or greater than `max_len`
            if min_len <= len(pieces) <= max_len:

                # add the BERT tokenized string as a comment line
                bert = '# bert = ' + ' '.join(pieces)
                sent = sent.split('\n', 2)
                sent = f'{sent[0]}\n{sent[1]}\n{bert}\n{sent[2]}'
                out.append(sent)

    # write the cleaned text to file
    with open(in_corpus_fn, 'w') as f:
        f.write('\n\n'.join(out))


def split_corpus(in_corpus_fn, train_fn, dev_fn, test_fn):
    '''Perform an 80-10-10 split on `in_corpus_fn`.'''
    with open(in_corpus_fn, 'rb') as f:
        text = SENT_RE.findall(f.read().decode('utf-8'))

    random.shuffle(text)

    N = len(text)
    i = int(N * 0.8)
    j = int(N * 0.9)

    train = text[:i]
    dev = text[i:j]
    test = text[j:]

    with open(train_fn, 'w') as f:
        f.write('\n\n'.join(train))

    with open(dev_fn, 'w') as f:
        f.write('\n\n'.join(dev))

    with open(test_fn, 'w') as f:
        f.write('\n\n'.join(test))


def reduce_train_corpus(in_corpus_fn, out_corpus_fn=None, k=800, feats_fn=None,
                        min_freq=100):
    ''' '''
    new = []

    with open(in_corpus_fn, 'rb+') as f:
        text = f.read().decode('utf-8')
        sents = SENT_RE.findall(text)

    if feats_fn:
        with open(feats_fn, 'rb+') as f:
            features = re.findall(r'.+', f.read().decode('utf-8'))

        FEAT_RE = []

        for feat in features:
            if '=' in feat:
                feat, val = feat.split('=', 1)
                feat = rf'{feat}=(?:.+,)?{val}(?:,.+)?'

                if len(re.findall(feat, text)) <= min_freq:
                    FEAT_RE.append(feat)

            elif text.count(feat) <= min_freq:
                FEAT_RE.append(feat)  # pos

        FEAT_RE = re.compile(r'|'.join(FEAT_RE))

        for sent in sents:
            if FEAT_RE.search(sent):
                new.append(sent)
                sents.remove(sent)
                k -= 1

        if k <= 0:
            raise ValueError('Uh oh. Extracted too many sentences.')

    new.extend(random.choices(sents, k=k))

    with open(out_corpus_fn or in_corpus_fn, 'w') as f:
        f.write('\n\n'.join(new))


def create_features(stats_xml_fn, feats_fn):
    ''' '''
    POS, FEATURES = [], set()
    tree = ET.parse(stats_xml_fn)
    root = tree.getroot()

    # extract parts of speech
    for tags in root.findall('tags'):
        for tag in tags:
            pos = tag.get('name')

            # exclude irrelevant parts of speech
            if pos not in EXCL_POS:
                POS.append(pos)

    # extract morphosyntactic features
    for feats in root.findall('feats'):
        for feat in feats:
            value = feat.get('value')
            feat = feat.get('name')

            # exclude features that are not so morphosyntactic-y
            if not EXCL_RE.search(feat):

                # split the values if the feature is multi-valued (e.g.,
                # 'Gender=Fem,Masc' becomes 'Gender=Fem' and 'Gender=Masc')
                for val in VALS_RE.split(value):
                    FEATURES.add(f'{feat}={val}')

    POS = '\n'.join(POS)
    FEATURES = '\n'.join(sorted(FEATURES))

    with open(feats_fn, 'w') as f:
        f.write(f'{POS}\n{FEATURES}')


def create_vocab(vocab_fn, train_fn, dev_fn, test_fn):
    ''' '''
    master_vocab = set()

    for corpus_fn in (test_fn, dev_fn, train_fn):
        with open(corpus_fn, 'rb') as f:
            vocab = set(get_tokens(f.read().decode('utf-8')))
            master_vocab.update(vocab)

    master_vocab = sorted(list(master_vocab))
    master_vocab = [f'{w} {int(w in vocab)}' for w in master_vocab]

    # write the vocabulary to file
    with open(vocab_fn, 'w') as f:
        f.write('\n'.join(master_vocab))


def get_tokens(text):
    ''' '''
    tokens = []

    if not isinstance(text, list):
        text = SENT_RE.findall(text)

    for sent in text:
        lines = (FEAT_RE.split(i) for i in sent.split('\n')[3:])

        try:
            while True:
                line = next(lines)
                idx, tok = line[0], line[1]
                tokens.append(tok)

                if '-' in idx:  # multiword token
                    i, j = (int(_) for _ in idx.split('-'))
                    for _ in range(i, j + 1):
                        next(lines)

        except StopIteration:
            continue

    return tokens


if __name__ == '__main__':

    # command things
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang')
    parser.add_argument('-s', '--split', action='store_true')
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-r', '--reduce', action='store_true')
    parser.add_argument('-k', '--k', type=int, default=800)
    parser.add_argument('-v', '--vocab', action='store_true')
    parser.add_argument('-f', '--features', action='store_true')
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--min_freq', type=int, default=100)
    parser.add_argument('--configs', default='configs.json')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    # load configs
    with open(args.configs, 'r') as f:
        configs = json.load(f)

    # determine the language
    try:
        config = configs[args.lang]

    except KeyError:
        raise ValueError(
            "Please specify a valid '--lang' argument:\n\t" +
            '\n\t'.join(configs.keys())
            )

    # retrieve the corpus filenames, features, etc.
    train_fn = config.get('train_fn')
    dev_fn = config.get('dev_fn')
    test_fn = config.get('test_fn')
    base_fn = config.get('base_fn', train_fn)

    # seed things for reproducibility
    random.seed(args.seed)

    # create a feature file
    if args.features:
        create_features(
            stats_xml_fn=config['stats_xml_fn'],
            feats_fn=config['data_params']['feats_fn'],
            )

    # clean the corpus
    if args.clean:
        for fn in (base_fn, dev_fn, test_fn):
            try:
                clean_corpus(
                    in_corpus_fn=fn,
                    min_len=args.min_len,
                    max_len=args.max_len,
                    )

            except (FileNotFoundError, TypeError):
                continue

    # perform an 80-10-10 split on the corpus
    if args.split:
        split_corpus(
            in_corpus_fn=base_fn,
            train_fn=train_fn,
            dev_fn=dev_fn,
            test_fn=test_fn,
            )

    # reduce the training set
    if args.reduce:
        reduce_train_corpus(
            in_corpus_fn=base_fn,
            out_corpus_fn=train_fn,
            k=args.k,
            feats_fn=config['data_params']['feats_fn'],
            min_freq=args.min_freq,
            )

# create a vocabulary file
if args.vocab:
    create_vocab(
        vocab_fn=config['data_params']['vocab_fn'],
        train_fn=train_fn,
        dev_fn=dev_fn,
        test_fn=test_fn,
        )
