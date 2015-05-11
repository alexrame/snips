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

##we retrieve the data in a useful way
trX,trY,teX,teY = getData(oh=0)
dtrain = xgb.DMatrix(trX, label = trY)
dtest = xgb.DMatrix(teX, label = teY)

##we will try different set of parameters inside this grid
hyperparams_grid = {
    'n_round' : 800,
    'early_stopping' : 100,
    'bst:max_depth' : scipy.stats.randint(10,30),
    'bst:eta' : scipy.stats.uniform(0.01,0.2),
    'bst:subsample' : scipy.stats.uniform(0.5,0.5),
    'bst:colsample_bytree' : scipy.stats.uniform(0.7,0.3)
    }
hyperparams_grid

##function that enables to test a set of parameters
def evaluate_hyperparams(hyperparams):
    def get_model_results(mdl):
            res = {}
            #res['ntrees'] = mdl.best_iteration+1
            #res['score'] = mdl.best_score
            res['best_ntrees'] = mdl.best_iteration+1
            res['best_score'] = mdl.best_score
            return res
    def get_results_df():
        if os.path.isfile('results/results_df_last.pkl'):
            return pd.io.pickle.read_pickle('results/results_df_last.pkl')
        else:
            return pd.DataFrame()

    n_round = hyperparams['n_round'] ## number of trees
    early_stopping = hyperparams['early_stopping'] ##if the prediction does not increase during a certain number of iterations, the algo stops
    plst = [
            ('bst:max_depth', hyperparams['bst:max_depth']), ##maximum depth
            ('objective', 'multi:softprob'), ##objective function
            ('silent', 1),
            ('bst:eta', hyperparams['bst:eta']), ##learning rate
            ('bst:subsample', hyperparams['bst:subsample']), ##data subsample
            ('bst:colsample_bytree', hyperparams['bst:colsample_bytree']), ##feature subsample
            ('num_class', 4),
    ]
    
    evallist  = [(dtrain,'train'), (dtest,'test')]
    mdl = xgb.train(plst, dtrain, 
                    num_boost_round = n_round, 
                    evals = evallist,
                    early_stopping_rounds = early_stopping) ## the actual training of our model
    
    line_dict = get_model_results(mdl)
    line_dict.update(hyperparams)
    line_dict = {k:[v] for k,v in line_dict.iteritems()}
    line_df = pd.DataFrame(line_dict)
    results_df = get_results_df()
    results_df = results_df.append(line_df, ignore_index = True)
    results_df.to_pickle('results/results_df_last.pkl') ##save the result of this set of parameters
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

##number of set of parameters to be tested   
N = 20

for k in xrange(N):
    print '--------------------------'
    print 'Evaluating hyperparameters'
    print k
    hp = {k:draw(v) for k, v in hyperparams_grid.iteritems()} ##randomly choose the set of parameters
    print hp
    evaluate_hyperparams(hp)