'''
This files interfaces with the data, including obtaining and caching the BERT
representations of the UD data.
'''
import argparse
import json
import random
import re

import numpy as np
import torch
import transformers as hf

from hashlib import sha1
from itertools import chain

from torch.utils.data import Dataset
from transformers import BertModel

from corpus import get_tokens, FEAT_RE, SENT_RE


class CoNLLUData(Dataset):

    def __init__(self, corpus_fn, feats_fn, bert_fn, vocab_fn=None, layer=12,
                 lang=None, control=False, aggregate='sum', cache=True):
        self.bert_fn = f'{bert_fn}_{layer:02}-'
        self.bert_fn += re.search(r'train|dev|test', corpus_fn).group(0)
        self.bert_fn += '.pt'
        self.layer = layer
        self.lang = lang

        # load features -------------------------------------------------------

        with open(feats_fn, 'rb+') as f:
            features = re.findall(r'.+', f.read().decode('utf-8'))

        self.features = []
        self.feat_idx = {}
        self.idx_feat = {}
        self.n_features = 0

        for feat in features:
            self.features.append(feat)
            self.feat_idx[feat] = self.n_features
            self.idx_feat[self.n_features] = feat
            self.n_features += 1

        # determine any OOV words ---------------------------------------------

        if 'train' in corpus_fn:
            self.oov = []

        else:
            try:
                with open(vocab_fn, 'rb+') as f:
                    self.oov = \
                        re.findall(r'.+(?= 0)', f.read().decode('utf-8'))
                    self.oov.sort()

            except FileNotFoundError:
                raise ValueError('Need a vocab file to get OOV words.')

        # load corpus and vectorize gold targets ------------------------------

        with open(corpus_fn, 'rb+') as f:
            text = SENT_RE.findall(f.read().decode('utf-8'))

        samples, init_mask = self.extract_gold(text)

        # obtain BERT representations of the inputs ---------------------------

        if cache:
            self.cache(samples)

            # obtain a single input representation for each word (i.e., UD
            # token), using the `aggregate` strategy
            self.aggregation_strategy = aggregate
            getattr(self, f'get_{aggregate}_reps')(init_mask)

        # create control data -------------------------------------------------

        if control:
            assert vocab_fn, "Need a vocab file to create control data."

            # create an effectively unique identifier given the set of features
            # to ensure that the same control dataset is used across
            # experiments that share features
            feats_id = sha1(
                str(sorted([i for i in self.features])).encode('utf-8')
                ).hexdigest()[:16]
            self.control_fn = f'{vocab_fn}-ctrl-{feats_id}.pt'

            try:
                self.create_control(vocab_fn, text)

            except FileNotFoundError:
                raise ValueError('Need a vocab file to create control data.')

        # default to gold labels ----------------------------------------------

        self.use_gold()

    def __len__(self):
        '''Return the total number of samples.'''
        return len(self.reps)

    def __getitem__(self, idx):
        '''Generate one sample.'''
        return (self.reps[idx], self.labels[idx])

    def extract_gold(self, text):
        '''Extract and vectorize the gold data.'''
        self.ud_tokens = []
        self.wp_tokens = []
        init_idxs = []
        mwt_idxs = []
        oov_idxs = []
        samples = []
        gold = []
        ud_widx = 0
        wp_widx = 0

        # track the number of word types that appear with each feature
        self.gold_feat_word_types = {f: set() for f in self.features}

        for sent in text:
            _, _, sent, TOKENS = sent.split('\n', 3)
            sent = re.search(r'(?<=# bert = ).+', sent).group().split()
            pieces = iter(sent)
            tokens = (FEAT_RE.split(i) for i in TOKENS.split('\n'))

            try:
                while True:
                    line = next(tokens)
                    idx, tok = line[0], line[1]
                    self.ud_tokens.append(tok)

                    if '-' in idx:  # multiword token
                        i, j = (int(_) for _ in idx.split('-'))
                        features = chain(*(next(tokens)[3:-4] for _ in range(i, j + 1)))  # noqa
                        mwt_idxs.append(ud_widx)

                    else:  # simplex
                        features = line[3:-4]

                    if tok in self.oov:
                        oov_idxs.append(ud_widx)

                    gold.append(self.vectorize_target(features, tok))
                    init_idxs.append(wp_widx)

                    # align the UD tokens to the BERT wordpieces to ascertain
                    # the first subword token of each word form
                    n = 1
                    tok = tok.replace(' ', '')
                    wp1 = next(pieces)
                    wp2 = [wp1, ]

                    try:
                        while tok != wp1 and wp1 != '[UNK]':
                            wp2.append(next(pieces))
                            wp1 += wp2[-1].replace('##', '', 1)
                            n += 1

                        ud_widx += 1
                        wp_widx += n
                        self.wp_tokens.append(' '.join(wp2))

                    except StopIteration:
                        raise RuntimeError('Failed to align UD~BERT tokens.')

            except StopIteration:
                samples.append(sent)

        init_mask = torch.zeros(wp_widx)
        init_mask[init_idxs] = 1
        self.mwt_mask = torch.zeros(ud_widx)
        self.mwt_mask[mwt_idxs] = 1
        self.oov_mask = torch.zeros(ud_widx)
        self.oov_mask[oov_idxs] = 1
        self.gold = torch.stack(gold)

        self.gold_feat_word_types = \
            [len(v) for v in self.gold_feat_word_types.values()]

        return samples, init_mask

    def vectorize_target(self, features, tok):
        '''Covert `feautures` into a multi-label (multi-hot encoded) vector.'''
        vec = torch.zeros(self.n_features)

        for feat in features:

            try:
                vec[self.feat_idx[feat]] = 1
                self.gold_feat_word_types[feat].add(tok)

            except KeyError:

                # dynamically handle ambiguous features (e.g., map
                # 'Gender=Fem,Masc' to 'Gender=Fem' and 'Gender=Masc')
                try:
                    parts = re.findall(r'[^=,]+', feat)
                    attr = parts[0]  # TODO: RENAME feat=value

                    for val in parts[1:]:
                        feat = f'{attr}={val}'
                        vec[self.feat_idx[feat]] = 1
                        self.gold_feat_word_types[feat].add(tok)

                except (IndexError, KeyError):
                    continue

        return vec

    def create_control(self, vocab_fn, text, thres=0.001):  # TODO
        '''Create multi-label control targets.'''
        with open(vocab_fn, 'rb+') as f:
            vocab = re.findall(r'.+(?= \d)', f.read().decode('utf-8'))

        # create word-level vocabulary
        self.word2idx = {w: i for i, w in enumerate(vocab)}

        try:
            control = torch.load(self.control_fn)

        except FileNotFoundError:
            W = len(self.word2idx)

            # get the distribution of features
            weights = self.gold.mean(0)
            weights[np.where(weights < thres)] = thres
            weights = weights.tolist()

            # create a random control vector for each word in the vocabulary
            # based on the distribution of features
            control = torch.zeros((W, self.n_features))

            for i in range(self.n_features):
                p = weights[i]
                control[:, i] = \
                    torch.tensor(np.random.choice(2, W, p=(1 - p, p)))

            torch.save(control, self.control_fn)

        # obtain the token indices
        idxs = [self.word2idx[t] for t in get_tokens(text)]

        # label the sentences with the control target vectors
        self.fake = torch.stack([control[i] for i in idxs])

        # track the number of word types that appear with each feature
        self.ctrl_feat_word_types = control[list(set(idxs))].sum(0).int()

    def cache(self, samples):
        ''' '''
        try:
            self.reps = torch.load(self.bert_fn)

        except FileNotFoundError:

            # cpu or gpu? that is the question
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # noqa
            print(f'Using {device}!')

            # load tokenizer
            self.tokenizer = hf.BertTokenizer.from_pretrained(
                'bert-base-multilingual-cased',
                do_lower_case=False,
                do_basic_tokenize=False,
                )

            # instantiate a pretrained multilingual bert
            model = BertModel.from_pretrained(
                'bert-base-multilingual-cased',
                output_hidden_states=True,
                ).to(device)

            # get some frozen CWRs
            for param in model.parameters():
                param.requires_grad = False

            model.eval()

            with torch.no_grad():
                self.reps = []

                # collect the contextualized bert representations sentence by
                # sentence from the desired layer
                for sample in samples:
                    enc = self.tokenizer.encode_plus(
                        sample,
                        return_token_type_ids=False,
                        return_attention_mask=False,
                        return_tensors='pt',
                        )['input_ids'].to(device)
                    enc = model(enc)[2][self.layer].to('cpu')
                    enc = enc.squeeze(0)[1:-1]  # excl. [CLS] and [SEP]
                    self.reps.append(enc)

                # cache the representations
                self.reps = torch.cat(self.reps)
                torch.save(self.reps, self.bert_fn)

    def get_init_reps(self, init_mask):
        '''Restrict the inputs to word-initial representations.'''
        init = np.where(init_mask)
        self.reps = self.reps[init]

    def get_fin_reps(self, init_mask):
        '''Restrict the inputs to word-final representations.'''
        init = np.where(init_mask)[0]
        fin = (init - 1)[1:].tolist() + [-1, ]
        self.reps = self.reps[fin]

    def get_sum_reps(self, init_mask):
        '''Return the sum of each word's subword representations.'''
        init = np.where(init_mask)[0].tolist()
        reps = []

        for i, j in zip(init, init[1:] + [None, ]):
            reps.append(self.reps[i:j].sum(0))

        self.reps = reps

    def get_mean_reps(self, init_mask):
        '''Return the mean of each word's subword representations.'''
        init = np.where(init_mask)[0].tolist()
        reps = []

        for i, j in zip(init, init[1:] + [None, ]):
            reps.append(self.reps[i:j].mean(0))

        self.reps = reps

    @property
    def simplex_mask(self):
        '''Return a mask of simplex tokens.'''
        return 1 - self.mwt_mask

    @property
    def vocab_mask(self):
        '''Return a mask of in-vocabulary tokens.'''
        return 1 - self.oov_mask

    @property
    def simplex_oov_mask(self):
        '''Return a mask of OOV simplex tokens.'''
        return self.simplex_mask * self.oov_mask

    @property
    def simplex_vocab_mask(self):
        '''Return a mask of in-vocabulary simplex tokens.'''
        return self.simplex_mask * self.vocab_mask

    @property
    def mwt_oov_mask(self):
        '''Return a mask of OOV multiword tokens.'''
        return self.mwt_mask * self.oov_mask

    @property
    def mwt_vocab_mask(self):
        '''Return a mask of in-vocabulary multiword tokens.'''
        return self.mwt_mask * self.vocab_mask

    def use_gold(self):
        '''Switch the dataset to return gold labels when sampling.'''
        self.labels = self.gold
        self.feat_word_types = self.gold_feat_word_types
        self.mode = 'gold'

    def use_control(self):
        '''Switch the dataset to return control labels when sampling.'''
        self.labels = self.fake
        self.feat_word_types = self.ctrl_feat_word_types
        self.mode = 'control'

    def count_features(self):
        '''Print the frequencies of each feature in the target labels.

        The format is as follows for each feature:
            <feat>,<support>,<type freq>

        where:
            - <support> is the # of tokens that appear with that feature
            - <type freq> is the # of word types that appear with that feature
        '''
        print('feature,support,type_count')

        counts = self.labels.reshape((-1, self.n_features)).sum(0).int()

        for feat, s, t in zip(self.features, counts, self.feat_word_types):
            print(feat, int(s), int(t), sep=',')


if __name__ == '__main__':

    # command things
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', required=True)
    parser.add_argument('-p', '--probe_layer', type=int, default=12)
    parser.add_argument('-i', '--index', action='store_true')
    parser.add_argument('-f', '--features', action='store_true')
    parser.add_argument('-d', '--dev', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-c', '--control', action='store_true')
    parser.add_argument('-m', '--transfer', type=str)
    parser.add_argument('-a', '--aggregate', default='sum',
                        choices=('init', 'fin', 'sum', 'mean'))
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--configs', default='configs.json')
    args = parser.parse_args()

    # seed things for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load configs
    with open(args.configs, 'r') as f:
        configs = json.load(f)

    # determine language/corpora
    try:
        config = configs[args.lang]
        train_fn = config['train_fn']
        dev_fn = config['dev_fn']
        data_params = config['data_params']

        if args.transfer:
            config = configs[args.transfer]

        test_fn = config['test_fn']
        test_data_params = config['data_params']
        test_data_params['feats_fn'] = data_params['feats_fn']

    except KeyError:
        raise ValueError(
            "Please specify a valid '--lang' argument:\n\t" +
            '\n\t'.join(configs.keys())
            )

    data_params['layer'] = test_data_params['layer'] = args.probe_layer
    data_params['control'] = test_data_params['control'] = args.control
    data_params['aggregate'] = test_data_params['aggregate'] = args.aggregate

    # cache the BERT representations (if they haven't already been cached)
    if args.cache:
        for corpus_fn in (train_fn, dev_fn, test_fn):
            CoNLLUData(
                corpus_fn=corpus_fn,
                **test_data_params if corpus_fn == test_fn else data_params,
                )

    # test dataset creation and indexing
    if args.index:
        for corpus_fn in (train_fn, dev_fn, test_fn):
            ds = CoNLLUData(
                corpus_fn=corpus_fn,
                **test_data_params if corpus_fn == test_fn else data_params,
                )

            if args.control:
                ds.use_control()

            for i in range(len(ds)):
                ds[i]

    # count the word types and tokens per feature in the specified dataset
    if args.features:
        corpus_fn = dev_fn if args.dev else test_fn if args.test else train_fn
        ds = CoNLLUData(
            corpus_fn=corpus_fn,
            cache=False,
            **test_data_params if corpus_fn == test_fn else data_params,
            )

        if args.control:
            ds.use_control()

        ds.count_features()
