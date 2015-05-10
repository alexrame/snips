import numpy as np
from load_new import getData
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
import pickle
import os.path
import scipy
import random

trX,trY,teX,teY = getData(oh=0)

dtrain = xgb.DMatrix(trX, label = trY)
dtest = xgb.DMatrix(teX, label = teY)

hyperparams_grid = {
    'n_round' : 800,
    'early_stopping' : 100,
    'bst:max_depth' : scipy.stats.randint(10,30),
    'bst:eta' : scipy.stats.uniform(0.01,0.2),
    'bst:subsample' : scipy.stats.uniform(0.5,0.5),
    'bst:colsample_bytree' : scipy.stats.uniform(0.7,0.3)
    }
hyperparams_grid

def evaluate_hyperparams(hyperparams):
    def get_model_results(mdl):
            res = {}
            res['ntrees'] = mdl.best_iteration+1
            res['score'] = mdl.best_score
            res['best_ntrees'] = mdl.best_iteration+1
            res['best_score'] = mdl.best_score
            return res
    def get_results_df():
        if os.path.isfile('results/results_df_last.pkl'):
            return pd.io.pickle.read_pickle('results/results_df_last.pkl')
        else:
            return pd.DataFrame()

    n_round = hyperparams['n_round']
    early_stopping = hyperparams['early_stopping']
    plst = [
            ('bst:max_depth', hyperparams['bst:max_depth']),
            ('objective', 'multi:softprob'),
            ('silent', 1),
            ('bst:eta', hyperparams['bst:eta']),
            ('bst:subsample', hyperparams['bst:subsample']),
            ('bst:colsample_bytree', hyperparams['bst:colsample_bytree']),
            ('num_class', 4),
    ]
    evallist  = [(dtrain,'train'), (dtest,'test')]
    mdl = xgb.train(plst, dtrain, 
                    num_boost_round = n_round, 
                    evals = evallist,
                    early_stopping_rounds = early_stopping)
    line_dict = get_model_results(mdl)
    line_dict.update(hyperparams)
    line_dict = {k:[v] for k,v in line_dict.iteritems()}
    line_df = pd.DataFrame(line_dict)
    results_df = get_results_df()
    results_df = results_df.append(line_df, ignore_index = True)
    results_df.to_pickle('results/results_df_last.pkl')
    results_df.to_pickle('results/old/results_df_%s.pkl' % (len(results_df)-1))
    file = open("results/results_last.txt", "w")
    file.write(str(results_df))
    file.close()
    file = open("results/old/results_%s.txt" % (len(results_df)-1), "w")
    file.write(str(results_df))
    file.close()
    
def draw(v):
    if hasattr(v, 'rvs'):
        return v.rvs()
    elif type(v) is list:
        return random.choice(v)
    else:
        return v

N = 15

for k in xrange(N):
    print '--------------------------'
    print 'Evaluating hyperparameters'
    print k
    hp = {k:draw(v) for k, v in hyperparams_grid.iteritems()}
    print hp
    evaluate_hyperparams(hp)