'''
This file contains the code for training a MorphBERT experiment.
'''
import warnings

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch

from torch import nn
from torch.optim import Adam

# cpu or gpu? that is the question
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MorphBERT:

    def __init__(self, features, model_fn, epochs=50):
        self.features = features
        self.n_features = len(features)
        self.model_fn = model_fn
        self.epochs = epochs

        # our mighty morpho probe!
        self.probe = nn.Linear(768, self.n_features).to(device)

        # let's train this thing
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(params=self.probe.parameters())

    def fit_train_thing(self, train_batches, val_batches, log_interval=10):
        ''' '''
        n_batches = len(train_batches)
        best_loss = float('inf')
        best_epoch = 0

        # da train lewp
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            interval_loss = 0.0

            # turn on training mode
            self.probe.train()

            for j, (inputs, labels) in enumerate(train_batches, start=1):
                inputs, labels = inputs.to(device), labels.to(device)

                # clear gradients from the last step
                self.optimizer.zero_grad()

                # push the representations thru bert
                # shape: (batch_size, 768)
                outputs = self.probe(inputs)

                # calculate the loss
                loss = self.criterion(outputs, labels)

                # calculate the gradient
                loss.backward()

                # update model parameters
                self.optimizer.step()

                # update the training and interval losses
                train_loss += loss.item()
                interval_loss += loss.item()

                # print mini-batch training progress
                if j % log_interval == 0:
                    interval_loss /= log_interval
                    print(f'| epoch {epoch} | batch {j}/{n_batches} | loss {interval_loss:.6f}')  # noqa
                    interval_loss = 0.0

            # take the average training loss across the batches
            train_loss /= n_batches

            # get the model's loss and predictions on the validation set
            val_loss, y_pred = self.evaluate(val_batches)

            # print epoch and losses
            print(f'| epoch {epoch} | train loss {train_loss:.6f} | val loss {val_loss:.6f}')  # noqa

            # save the model with the best validation loss
            if val_loss < best_loss:
                self.save()
                best_loss = val_loss
                best_epoch = epoch
                print('**** best loss')

            # print validation precision, recall, and F1
            print(self.eval_morph(val_batches.dataset, y_pred))

        # print the epoch with the best loss
        print(f'**** best loss @ epoch {best_epoch}\n\n')

        # load the model with the best validation loss
        self.load()

    def evaluate(self, batches):
        '''Return the model's loss and predicted vectors for `batches`.'''
        # turn on evaluation mode
        self.probe.eval()

        # accrue outputs for morphological analysis
        y_pred = []

        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in batches:
                inputs, labels = inputs.to(device), labels.to(device)

                # push the representations thru bert
                # shape: (batch_size, 768)
                outputs = self.probe(inputs)

                # update the validation loss
                val_loss += self.criterion(outputs, labels).item()

                y_pred.append(outputs)

        # take the average validation loss across the batches
        val_loss /= len(batches)

        # shape: (n_sentences, seq_len, n_features)
        y_pred = (torch.cat(y_pred, 0) >= 0.5).int().cpu()  # binarize

        return val_loss, y_pred

    def eval_morph(self, dataset, y_pred, mask=None):
        '''Calculate precision, recall, and F1 across the predictions.

        Predictions are done at the word level.
        '''
        y_true = dataset.labels

        if mask:
            idxs = np.where(getattr(dataset, f'{mask}_mask'))
            y_true = y_true[idxs]
            y_pred = y_pred[idxs]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            labels = [1, ] if y_true.size(1) == 1 else None

            # precision, recall, and f1
            report = metrics.classification_report(
                y_true,
                y_pred,
                target_names=self.features,
                labels=labels,
                )

            # hamming loss
            ham = round(metrics.hamming_loss(y_true, y_pred), 4)

            # subset accuracy
            acc = round(metrics.accuracy_score(y_true, y_pred), 4)

        return f'\n{report}\n hamming loss\t   {ham}\n     accuracy\t   {acc}\n\n'  # noqa

    def save(self):
        '''Save the linear probe to `model_fn`.'''
        torch.save(self.probe.state_dict(), f'{self.model_fn}.pt')

    def load(self):
        '''Load the linear probe from `model_fn`.'''
        classifier = torch.load(f'{self.model_fn}.pt', map_location=device)
        self.probe.load_state_dict(classifier)


def create_infer_df(dataset, y, csv_fn=None):
    '''Convert vectorized labels into a readable CSV.'''
    columns = ['lang', 'token', 'bert_wp'] + dataset.features + ['mwt', 'oov']
    pred = np.array(y)
    true = np.array(dataset.labels)

    # indicate true negatives (0), true positives (2), false negatives (-1),
    # and false positives (1)
    features = np.zeros(pred.shape)
    features[np.where(pred + true == 2)] = 2
    features[np.where(pred * 2 - true == 2)] = 1
    features[np.where(true * 2 - pred == 2)] = -1

    data = np.concatenate((
        np.zeros((len(dataset), 3)),           # lang, token, bert_wp
        features,                              # features
        np.expand_dims(dataset.mwt_mask, -1),  # multiword tokens
        np.expand_dims(dataset.oov_mask, -1),  # out-of-vocab tokens
        ), -1)

    # create the initial data frame
    df = pd.DataFrame(data=data, columns=columns).astype(int)

    # add the language(s) to the data frame rows
    df.lang = dataset.lang

    # add the UD tokens and BERT word pieces to the data frame rows
    df.token = dataset.ud_tokens
    df.bert_wp = dataset.wp_tokens

    # write the data frame to file
    if csv_fn:
        df.to_csv(csv_fn, index=False)

    return df
