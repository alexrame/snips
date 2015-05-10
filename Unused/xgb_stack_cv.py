import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import os.path
import scipy
import random
from load_new import getData

#from loadfft import getData
from sklearn import cross_validation
from sklearn.ensemble.base import BaseEnsemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from blend_nn import NeuralNet
from sklearn.cross_validation import StratifiedKFold

import math



class ModifiedXGBClassifier(xgb.XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, 
                 silent=True, objective="reg:linear", max_features=1, subsample = 1):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.silent = silent
        self.n_estimators = n_estimators
        self.objective = objective
        self.max_features = max_features
        self.subsample = subsample
        self._Booster = xgb.Booster()
        
    def get_params(self, deep=True):
        return {'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'silent': self.silent,
                'objective': self.objective,
                'max_features' : self.max_features,
                'subsample' : self.subsample,
                'num_class':4
                }
        
    def get_xgb_params(self):
        return {'eta': self.learning_rate,
                'max_depth': self.max_depth,
                'silent': 1 if self.silent else 0,
                'objective': self.objective,
                'bst:subsample': self.subsample,
                'bst:colsample_bytree': self.max_features,
                'num_class':4
                }
    
class BlendedModel(BaseEnsemble):
    def __init__(self, models=[]):
        self.models = models
    
    def fit(self, X, y):
        for model in self.models:
            print 'Training model :'
            print model.get_params()
            model.fit(X, y)                
        return self
    

if __name__ == '__main__': 
    results = pd.io.pickle.read_pickle('results/results_df_last.pkl')
    models_to_use = results[(results['best_score']<0.07) | ((results['best_score']<0.09)&(results['bst:eta']>0.3))]
    models_to_use=models_to_use.sort(['best_score'])[:5]
    models = []
    for j,row in enumerate(models_to_use.iterrows()):
            print j
            hyperparams = dict(row[1])
            models.append(ModifiedXGBClassifier(
                                    max_depth=hyperparams['bst:max_depth'], 
                                    learning_rate=hyperparams['bst:eta'], 
                                    n_estimators=int(hyperparams['best_ntrees']),
                                    max_features=hyperparams['bst:colsample_bytree'],
                                    subsample=hyperparams['bst:subsample'],
                                    silent=True, 
                                    objective='multi:softprob')
                          )
    nb_fold=4
    dtrX,dtrY,teX,teY = getData(prop=1./6,oh=0)
    skf = StratifiedKFold(dtrY, nb_fold)
    
    blend_train = np.zeros((dtrX.shape[0], len(models),4)) # Number of training data x Number of classifiers
    blend_test=np.zeros((teX.shape[0], len(models),4))

    for j,(train, val) in enumerate(skf):

            #print j,train, test
        vaX=dtrX[val,]
        trX=dtrX[train,]
        vaY=dtrY[val,]
        trY=dtrY[train,]


        bmdl = BlendedModel(models)
        bmdl.fit(trX,trY)

        print("score models")
        for m,model in enumerate(models):
            print(np.mean(np.argmax(model.predict_proba(vaX), axis=1) == vaY))        
            blend_train[val, m,:] = model.predict_proba(vaX)
            # Take the mean of the predictions of the cross validation set

    bmdl = BlendedModel(models)
    bmdl.fit(dtrX,dtrY)
    print("score models")
    for m,model in enumerate(models):
        print(np.mean(np.argmax(model.predict_proba(teX), axis=1) == teY))        
        blend_test[:, m,:] = model.predict_proba(teX)

    # Start blending!
    blend_train=[np.array(blend_train[x,:,:]).flatten() for x in range(len(dtrX))]
    blend_test=[np.array(blend_test[x,:,:]).flatten() for x in range(len(teX))]
    
    log=LogisticRegression()
    log.fit(blend_train, dtrY)
    print(np.mean(np.argmax(log.predict_proba(blend_train), axis=1) == dtrY))

    print(np.mean(np.argmax(log.predict_proba(blend_test), axis=1) == teY))


