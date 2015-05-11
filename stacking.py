import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import os.path
import scipy
import random

from sklearn import cross_validation
from sklearn.ensemble.base import BaseEnsemble
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
import math

from neuralnet import NeuralNet
from load_new import getData

##Gradient boosting trees from xgboost slightly modified
class ModifiedXGBClassifier(xgb.XGBClassifier):
    def __init__(self, max_depth=20, learning_rate=0.1, n_estimators=300, 
                 silent=True, objective='multi:softprob', max_features=0.3, subsample = 0.5):
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

##classifier with different kind of stacking classifier
##for each stacking classifier, we define a fit and a predict method
class BlendedModel(BaseEnsemble):
    def __init__(self, models=[], blending='average',nbFeatures=4):
        self.models = models
        self.blending = blending
        self.logR = LogisticRegression(C=10)#,multi_class='multinomial',solver='lbfgs', max_iter=10000)
        self.logRT= LogisticRegression(C=10)#,multi_class='multinomial',solver='lbfgs', max_iter=10000)
        self.nn=NeuralNet(nbFeatures) 
        self.XGB=ModifiedXGBClassifier()
        if self.blending not in ['average', 'most_confident']:
            raise Exception('Wrong blending method')
    
    ##fit the stochastic gradient boosting trees classifier
    def fit(self, X, y): 
        for model in self.models:
            print 'Training model :'
            print model.get_params()
            model.fit(X, y)                
        return self
    
    ##predict the outputs according to the average of the classifier (or according an entropy based voting scheme that does not work well)
    def predict_proba(self, X):
        preds = np.array(
                    [model.predict_proba(X) for model in self.models]
                )
        if self.blending == 'average':
            return np.mean(preds , axis=0 )
        elif self.blending == 'most_confident':
            def dirac_weights(entropies):
                w = (entropies == np.min(entropies)).astype(float)
                return w/np.sum(w)
            def shannon_entropy(l):
                l = [min(max(1e-5,p),1-1e-5) for p in l]
                l = np.array(l)/sum(l)
                return sum([-p*math.log(p) for p in l])
            shannon_entropy_array = lambda l : np.apply_along_axis(shannon_entropy, 1, l)
            entropies = np.array([shannon_entropy_array(pred) for pred in preds])
            weights = np.apply_along_axis(dirac_weights, 0, entropies)
            return np.sum(np.multiply(weights.T, preds.T).T, axis = 0)
 
    
    ##fit the logistic regression stacking
    def fitLog(self,X,y,mod=0):
        if mod==0: ##without features engineering
            preds = np.array(
                        [model.predict_proba(X) for model in self.models]
                    )
            features=np.array([np.array([preds[j][i] for j in range(len(self.models))]).flatten() for i in range(len(X))])
            self.logR.fit(features, y)
        elif mod==1: ##with features engineering
            preds = np.array(
                        [model.predict_proba(X) for model in self.models]
                    )
            features=np.array([np.array([[math.log(preds[j][i][k]/(1-preds[j][i][k])) for k in range(4)] for j in range(len(self.models))]).flatten() for i in range(len(X))])
            self.logRT.fit(features, y)
        return self
    
  
    ##predict the outputs of the logistic regression stack
    def predict_Logproba(self, X,mod=0):
        if mod==0:
            preds = np.array(
                        [model.predict_proba(X) for model in self.models]
                    )
            features=np.array([np.array([preds[j][i] for j in range(len(self.models))]).flatten() for i in range(len(X))])
            preds=self.logR.predict_proba(features)
            return preds
        elif mod==1:
            preds = np.array(
                    [model.predict_proba(X) for model in self.models]
                )
            features=np.array([np.array([[math.log(preds[j][i][k]/(1-preds[j][i][k])) for k in range(4)] for j in range(len(self.models))]).flatten() for i in range(len(X))])
            preds=self.logRT.predict_proba(features)
            return preds
        
    ##I also try to use a gradient boosting as a stacking classifier, but it does not work well    
    def fitXGB(self,X,y):
        preds = np.array(
                        [model.predict_proba(X) for model in self.models]
                    )
        features=np.array([np.array([[math.log(preds[j][i][k]/(1-preds[j][i][k])) for k in range(4)] for j in range(len(self.models))]).flatten() for i in range(len(X))])
        #features= np.append(features, X, axis=1)
        self.XGB.fit(features,y)
        return self
    
    def predict_XGBproba(self,X):
        preds = np.array(
                        [model.predict_proba(X) for model in self.models]
                    )
        features=np.array([np.array([[math.log(preds[j][i][k]/(1-preds[j][i][k])) for k in range(4)] for j in range(len(self.models))]).flatten() for i in range(len(X))])
        #features= np.append(features, X, axis=1)
        return self.XGB.predict_proba(features)
  
    ##neural network stack classifier
    def fitNN(self,X,y,lambda1=0.00000001,lambda2=0.00005,new=0,teX=[],teY=[],lr=0.001):
        
        preds = np.array(
                        [model.predict_proba(X) for model in self.models]
                    )
        features=np.array([np.array([preds[j][i] for j in range(len(self.models))]).flatten() for i in range(len(X))]) 
        features= np.append(features, X, axis=1)
        
        if len(teX)>0:
            preds = np.array(
                        [model.predict_proba(teX) for model in self.models]
                    )
            featuresteX=np.array([np.array([preds[j][i] for j in range(len(self.models))]).flatten() for i in range(len(teX))])
            featuresteX= np.append(featuresteX, teX, axis=1)
        else:
            featuresteX=[]
            
        self.nn.fit(features,y,lambda1,lambda2,new,featuresteX,teY,lr)
        return self

    ##predict nn stack classifier
    def predict_NNproba(self,X):
        preds = np.array(
                        [model.predict_proba(X) for model in self.models]
                    )
        features=np.array([np.array([preds[j][i] for j in range(len(self.models))]).flatten() for i in range(len(X))])
        features= np.append(features, X, axis=1)
        return self.nn.predict_proba(features)