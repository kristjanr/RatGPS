
from hyperopt import hp
from hyperopt import tpe

from optimizer import HyperBoostOptimizer


def parameters():
    loc_para = {
        'window_len': hp.uniformint('window_len', 10, 200),
        'window_hop': hp.uniformint('window_hop', 5, 40),
        "skip": hp.choice('skip', [0]),
    }
    sents_param = loc_para.copy()
    sents_param.update({
        'max_empty_words': hp.uniformint('max_empty_words', 0, 10),
        'word_ordering': hp.choice('word_ordering', ['shuffle', 'sort']),
    })

    w2v_params = {
        'vector_size': hp.uniformint('vector_size', 150, 500), # maybe fine-tune later
        'window': hp.uniformint('window', 4, 20),
        'alpha': hp.normal('alpha', 0.025, 0.005),
        'shrink_windows': hp.choice('shrink_windows', [True, False]), # some experimental stuff
        'sg': hp.choice('sg', [0, 1]), # ?
    } # this is still an incomplete list of hyperparameters

    features_para = {
        'method': hp.choice('method', ['mean', 'concatenate'])
    }

    params = {
        'loc_para': loc_para,
        'sents_para': sents_param,
        'w2v_para': w2v_params,
        'features_para': {}, # we don't need this when we use LSTM
        'random_forest_para': {}, # We do not care about the RandomForest hyperparams since we will use LSTM anyway
    }

    return params


def optimize_param_space():
    params = parameters()
    obj = HyperBoostOptimizer(fn_name='linreg', space=params) # random_forest or linreg as fn_name
    opt = obj.process(algo=tpe.suggest, max_evals=1000)
    print(opt)


if __name__ == "__main__":
    optimize_param_space()


# {'features_para': {}, 'loc_para': {'skip': 0, 'window_hop': 18, 'window_len': 66}, 'random_forest_para': {}, 'sents_para': {'max_empty_words': 6, 'skip': 0, 'window_hop': 18, 'window_len': 66, 'word_ordering': 'shuffle'}, 'w2v_para': {'sg': 0, 'shrink_windows': True, 'vector_size': 250, 'window': 19}}