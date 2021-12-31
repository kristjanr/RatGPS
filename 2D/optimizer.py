import math
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from random import shuffle
from time import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from hyperopt import fmin, STATUS_OK
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from monitor import Monitor


def prepare_data():
    print('Reading data...')
    raw_spikes = pd.read_csv("https://drive.google.com/uc?id=1-IuXanc2RZICrzqEG7jTg7f_AN9K9Ufl&export=download",
                             header=None)
    raw_loc = pd.read_csv("https://drive.google.com/uc?id=1-XHfaaYkfNeQHYefY4uGX4r1q3AKGbic&export=download",
                          header=None)
    raw_loc.columns = ["x", "y"]
    return raw_spikes, raw_loc


class HyperBoostOptimizer(object):
    RANDOM_STATE = 42
    NFOLDS = 5

    def __init__(self, fn_name, space):
        self.fn = getattr(self, fn_name)
        self.space = space
        self.filename = f'{fn_name}_at_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        self.X, self.y = prepare_data()
        self.baseline_loss = self.find_baseline_loss() # comment out to save time while debugging
        self.monitor = Monitor(self.baseline_loss, self.filename)
        np.random.seed(self.RANDOM_STATE)

    def process(self, algo, max_evals, existing_trials_file=None):
        ts = time()
        result = fmin(
            fn=self.crossvalidate,
            space=self.space,
            algo=algo,
            max_evals=max_evals,
            early_stop_fn=self.monitor.inspect_result,
            trials_save_file=existing_trials_file if existing_trials_file else f'{self.filename}.bin'
        )
        te = time()
        print('hyperopt took: %2.4f sec' % (te - ts))
        return result

    def crossvalidate(self, para):
        print(f'Parameters:')
        print(para)
        print('Building sentences...')
        sents = build_sentences(self.X, **para['sents_para'])
        locations = get_locations(self.y, **para['loc_para'])
        oof_preds = np.zeros(locations.shape)

        folds = KFold(n_splits=self.NFOLDS, shuffle=True, random_state=self.RANDOM_STATE)
        for fold_, (train_index, valid_index) in enumerate(folds.split(sents, locations)):
            trn_x, trn_y = list(itemgetter(*train_index)(sents)), locations.iloc[train_index, :]
            val_x, val_y = list(itemgetter(*valid_index)(sents)), locations.iloc[valid_index, :]
            val_pred = self.train_predict(para, trn_x, trn_y, val_x)
            val_dists = np.sqrt((val_y['x'] - val_pred.T[0]) ** 2 + (val_y['y'] - val_pred.T[1]) ** 2)
            print(f'Fold {fold_} loss {np.mean(val_dists)}')
            oof_preds[valid_index, :] = val_pred

        preds = oof_preds.T

        dists = np.sqrt((locations['x'] - preds[0]) ** 2 + (locations['y'] - preds[1]) ** 2)
        loss = np.mean(dists)
        print(f'loss {loss}')
        return {'loss': loss, 'status': STATUS_OK}

    def train_predict(self, para, trn_sents, trn_y, val_x):
        try:
            w2v_model = Word2Vec(compute_loss=True, seed=self.RANDOM_STATE, workers=6, **para['w2v_para'])
            w2v_model.build_vocab(trn_sents)
            print('Training Word2Vec...')
            w2v_model.train(corpus_iterable=trn_sents, total_examples=len(trn_sents), epochs=20, compute_loss=True)

            print('Building features...')
            features_train = build_features(w2v_model, trn_sents, **para['features_para'])

            regressor_model = self.fn(para)
            print(f'Training {regressor_model.estimator.__class__}...')
            regressor_model.fit(features_train, trn_y)

            features_val = build_features(w2v_model, val_x, **para['features_para'])

            return regressor_model.predict(features_val)
        except Exception:
            pass

    def random_forest(self, para):
        return MultiOutputRegressor(RandomForestRegressor(
            random_state=self.RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=3,  # TODO: this needs to be in the hyperparams space too
            **para['random_forest_para']))

    def linreg(self, para):
        return MultiOutputRegressor(LinearRegression())

    def find_baseline_loss(self):
        print('Finding baseline loss...')
        baseline_loss = self.crossvalidate(defaultdict(dict))['loss']
        print(f'Baseline loss: {baseline_loss}')
        return baseline_loss


def build_sentences(df, window_len=10, window_hop=10, skip=0, max_empty_words=2, word_ordering="shuffle"):
    """
    builds sentence vectors from spiking data
    window_len: how many spiking timesteps are used for one sentence
    window_hop: how much to slide the window. if window_len==window_hop there is no overlap. if hop<len, there is overlap.
    max_empty_words: how many consecutive empty words ("_") to keep. if 0, they are removed alltogether. if -1, keep all of them
    word_ordering: "shuffle" or "sort". If several neurons spike in one timestep, how to order them. If the same set of neurons spiked, maybe we should have consistent ordering?
    skip: how many timesteps to skip from the beginning. maybe the rat is training at first and we should discard that data?
    """

    sents = []
    for w in get_windows(len(df), window_len, window_hop, skip):
        start, end, mid = w

        # process one sentence
        sent_words = []
        empty_seq_len = 0
        for j in range(start, end):
            row = df.iloc[j]

            if np.sum(row) == 0:
                empty_seq_len += 1

                if max_empty_words == -1 or max_empty_words >= empty_seq_len:
                    sent_words.append("_")

            else:
                empty_seq_len = 0

                one_spike = row[row == 1].index.tolist()
                two_spikes = row[row == 2].index.tolist()
                three_spikes = row[row == 3].index.tolist()
                four_spikes = row[row == 4].index.tolist()
                word = 4 * four_spikes + 3 * three_spikes + 2 * two_spikes + one_spike

                if word_ordering == "shuffle":
                    shuffle(word)
                else:
                    word.sort()

                sent_words += word

        if len(sent_words) == 0:
            sent_words = ["_"]

        sents.append(sent_words)
    return sents


def get_windows(df_len, window_len=10, window_hop=10, skip=0):
    num_windows = math.ceil((df_len - skip) / window_hop)
    answer = []
    for i in range(num_windows):
        start = skip + i * window_hop
        end = min(start + window_len, df_len - 1)
        mid = math.floor((start + end) / 2)
        answer.append((start, end, mid))

    return answer


def get_locations(df, window_len=10, window_hop=10, skip=0):
    idx = [False] * len(df)
    for w in get_windows(len(df), window_len, window_hop, skip):
        mid = w[2]
        idx[mid - 1] = True

    return df[idx]


def build_features(word2vec_model, sents, method="mean"):
    features = []
    for sent in sents:
        vecs = np.array([word2vec_model.wv[word] for word in sent if word2vec_model.wv.has_index_for(word)])
        if method == "mean":
            features.append(np.mean(vecs, axis=0))
        else:
            # concatenate instead
            features.append(vecs.flatten())

    if method != "mean":
        # right-pad every row with zeros so that every vector has the same length
        maxlen = max([len(row) for row in features])
        features = [np.append(row, [0] * (maxlen - len(row))) for row in features]
        features = np.array(features)
    return features
