'''
This file creates a wrapper around main.py, so that it's not necessary to
re-launch CUDA for each experiment (which typically takes longer than training
the models themselves).
'''
import random
import time as t

import numpy as np
import torch

from contextlib import redirect_stdout

from main import main


# seed things for reproducibility
seed = 5
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    print('Using a GPU!')
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

LAYERS = (0, 2, 4, 6, 8, 10, 12)


def train_bunch(lang, transfer=None, probe_layers=LAYERS,
                do_real=True, do_control=True):
    ''' '''
    base_fn = f'out/{lang}'

    if transfer:
        base_fn += f'_{transfer}'

    exp_fn = f'{base_fn}'

    for layer in probe_layers:
        exp_fn = f'{exp_fn}_{layer:02}'
        params = dict(lang=lang, transfer=transfer,
                      probe_layer=layer)

        if do_real:
            out_fn = f'{exp_fn}-sum.out'

            if not transfer:
                # real - train
                print(out_fn)
                time = t.time()
                with open(out_fn, 'w') as f:
                    with redirect_stdout(f):
                        main(**params)
                print(t.time() - time)

                with open(out_fn, 'a') as f:
                    with redirect_stdout(f):
                        # real - dev
                        main(evaluate=True, infer=True, **params)

            with open(out_fn, 'a') as f:
                with redirect_stdout(f):
                    # real - test
                    main(evaluate=True, infer=True, test=True, **params)

        if do_control:
            out_fn = f'{exp_fn}-ctrl-sum.out'

            if not transfer:
                # control - train
                print(out_fn)
                time = t.time()
                with open(out_fn, 'w') as f:
                    with redirect_stdout(f):
                        main(control=True, **params)
                print(t.time() - time)

                with open(out_fn, 'a') as f:
                    with redirect_stdout(f):
                        # control - dev
                        main(control=True, evaluate=True, infer=True, **params)  # noqa

            with open(out_fn, 'a') as f:
                with redirect_stdout(f):
                    # control - test
                    main(control=True, evaluate=True, infer=True, test=True, **params)  # noqa


if __name__ == '__main__':
    pass
