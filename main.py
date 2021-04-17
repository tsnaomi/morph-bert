'''
This file creates a wrapper around launching MorphBERT experiment, collecting
evaluation data from a particular experiment, etc.
'''
import argparse
import json
import os.path as op
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from data import CoNLLUData
from experiment import MorphBERT, create_infer_df


def main(lang, probe_layer=12, transfer=None, control=False, force_train=False,
         test=False, evaluate=False, infer=False, aggregate='sum', bsz=512,
         epochs=50, configs='configs.json', csv_dir='csv', model_dir='models'):
    # load configs
    with open(configs, 'r') as f:
        configs = json.load(f)

    # determine language/copora
    try:
        # train/dev config
        config = configs[lang]
        train_fn = config['train_fn']
        dev_fn = config['dev_fn']
        model_fn = config['model_fn']
        data_params = config['data_params']
        csv_fn = ('', '')

        if transfer:
            config = configs[transfer]

            if test:
                csv_fn = (model_fn, f'{lang[0]}_{lang[1]}')

        # test config
        test_fn = config['test_fn']
        test_data_params = config['data_params']
        test_data_params['feats_fn'] = data_params['feats_fn']

    except KeyError:
        raise ValueError(
            "Please specify a valid '--lang' argument:\n\t" +
            '\n\t'.join(configs.keys())
            )

    data_params['layer'] = test_data_params['layer'] = probe_layer
    data_params['control'] = test_data_params['control'] = control
    data_params['aggregate'] = test_data_params['aggregate'] = aggregate

    # modify `model_fn` to include the probe layer
    model_fn += f'_{probe_layer:02}'

    # create datasets and data generators
    train_data = CoNLLUData(corpus_fn=train_fn, **data_params)
    train_generator = DataLoader(dataset=train_data, batch_size=bsz)

    dev_data = CoNLLUData(corpus_fn=dev_fn, **data_params)
    dev_generator = DataLoader(dataset=dev_data, batch_size=bsz)

    if test:
        eval_header = '#### TEST ####\n'
        test_data = CoNLLUData(corpus_fn=test_fn, **test_data_params)
        test_generator = DataLoader(dataset=test_data, batch_size=bsz)

        # set the test set as the evaluation set
        eval_data = test_data
        eval_generator = test_generator

    else:
        eval_header = '#### DEV ####\n'

        # set the dev set as the evaluation set
        eval_data = dev_data
        eval_generator = dev_generator

    # switch to the control task
    if control:
        train_header = '#### TRAIN (CONTROL TASK) ####\n'
        model_fn += '-ctrl'
        train_data.use_control()
        eval_data.use_control()

    else:
        train_header = '#### TRAIN ####\n'

    # note the aggregation strategy
    model_fn += '-' + aggregate

    # instantiate the model
    hehe = MorphBERT(
        features=train_data.features,
        model_fn=op.join(model_dir, model_fn),
        epochs=epochs,
        )

    # train the probe (overwrite any existing probe files)
    if force_train:
        print(train_header)
        hehe.fit_train_thing(
            train_batches=train_generator,
            val_batches=dev_generator,
            )

    else:
        try:
            # load the pre-trained probe
            hehe.load()

        except FileNotFoundError:
            # train the probe (since it has not been trained before)
            print(train_header)
            hehe.fit_train_thing(
                train_batches=train_generator,
                val_batches=dev_generator,
                )

    if evaluate or infer:
        eval_loss, y_pred = hehe.evaluate(eval_generator)

        # get performance (e.g., P, R, F1) on the evaluation set
        if evaluate:
            print(eval_header)
            print(f'Evaluation loss: {eval_loss:.6f}\n\nAll:')
            print(hehe.eval_morph(eval_data, y_pred))
            print('In vocabulary:')
            print(hehe.eval_morph(eval_data, y_pred, mask='vocab'))
            print('Out of vocabulary:')
            print(hehe.eval_morph(eval_data, y_pred, mask='oov'))

            if eval_data.mwt_mask.sum():
                print('Simplex:')
                print(hehe.eval_morph(eval_data, y_pred, mask='simplex'))
                print('Multiword tokens:')
                print(hehe.eval_morph(eval_data, y_pred, mask='mwt'))

        # get predictions on the evaluation set and write them to file
        if infer:
            csv_fn = op.join(csv_dir, model_fn.replace(*csv_fn))
            csv_fn += '-test' if test else '-dev'
            create_infer_df(eval_data, y_pred, csv_fn + '.csv')


if __name__ == '__main__':

    # command things
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', required=True)
    parser.add_argument('-p', '--probe_layer', type=int, default=12)
    parser.add_argument('-m', '--transfer', type=str)
    parser.add_argument('-c', '--control', action='store_true')
    parser.add_argument('-f', '--force_train', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-i', '--infer', action='store_true')
    parser.add_argument('-a', '--aggregate', default='sum',
                        choices=('init', 'fin', 'sum', 'mean'))
    parser.add_argument('--bsz', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--configs', default='configs.json')
    parser.add_argument('--csv_dir', default='csv')
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    # seed things for reproducibility
    seed = args.seed
    delattr(args, 'seed')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        print('Using a GPU!')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(**vars(args))
