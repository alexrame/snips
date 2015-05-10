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
from stacking import ModifiedXGBClassifier
from stacking import BlendedModel

if __name__ == '__main__': 
    results = pd.io.pickle.read_pickle('results/results_df_last.pkl')
    models_to_use = results#[(results['best_score']<0.07) | ((results['best_score']<0.09)&(results['bst:eta']>0.3))]
    models_to_use=models_to_use.sort(['best_score'])[:4]
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

    nb_fold=3
    dtrX,dtrY,dteX,dteY = getData(prop=0,oh=0)
    skf = StratifiedKFold(dtrY, nb_fold)
    
    rModTe=np.zeros((len(models),nb_fold))
    rAveTe=np.zeros(nb_fold)
    rLogTe=np.zeros(nb_fold)
    rLogVa=np.zeros(nb_fold)
    rLog2Te=np.zeros(nb_fold)
    rLog2Va=np.zeros(nb_fold)
    rXGBTe=np.zeros(nb_fold)
    rXGBVa=np.zeros(nb_fold)    
    rNNTe=np.zeros(nb_fold)
    rNNVa=np.zeros(nb_fold)

        
    for j,(train, test) in enumerate(skf):
        if j==0:
        #print j,train, test
            vaX=dtrX[test,][:(len(test)/2),]
            teX=dtrX[test,][(len(test)/2):,]
            trX=dtrX[train,]
            vaY=dtrY[test][:(len(test)/2)]
            teY=dtrY[test][(len(test)/2):]
            trY=dtrY[train]


            bmdl = BlendedModel(models,nbFeatures=(len(models)*4+len(trX[0])))
            bmdl.fit(trX,trY)

            print("score models")
            for m,model in enumerate(models):
                print(np.mean(np.argmax(model.predict_proba(teX), axis=1) == teY))
                rModTe[m][j]=(np.mean(np.argmax(model.predict_proba(teX), axis=1) == teY))

            predictTe=bmdl.predict_proba(teX)
            scoreTe=np.mean(np.argmax(predictTe, axis=1) == teY) 
            print("score average of models")
            print(scoreTe)
            rAveTe[j]=scoreTe

            print("blend logistic regression")
            bmdl.fitLog(vaX,vaY)

            predictTe=bmdl.predict_Logproba(teX)
            scoreTe=np.mean(np.argmax(predictTe, axis=1) == teY) 
            print("score blend logistic regression")
            print(scoreTe)
            rLogTe[j]=scoreTe


            predictVa=bmdl.predict_Logproba(vaX,mod=0)
            scoreVa=np.mean(np.argmax(predictVa, axis=1) == vaY) 
            print("score blend logistic regression val test")
            print(scoreVa)
            rLogVa[j]=scoreVa
       
           
            print("blend logistic regression with log(x/(1-x)")
            bmdl.fitLog(vaX,vaY,mod=1)
            predictTe=bmdl.predict_Logproba(teX,mod=1)
            scoreTe=np.mean(np.argmax(predictTe, axis=1) == teY) 
            print("score blend logistic regression with log(x/(1-x)")
            rLog2Te[j]=scoreTe
            print(scoreTe)


            predictVa=bmdl.predict_Logproba(vaX,mod=1)
            scoreVa=np.mean(np.argmax(predictVa, axis=1) == vaY) 
            print("score blend logistic regression with log(x/(1-x) val test")
            print(scoreVa)
            rLog2Va[j]=scoreVa          
            
            print("blend XGB")
            bmdl.fitXGB(vaX,vaY)

            predictTe=bmdl.predict_XGBproba(teX)
            scoreTe=np.mean(np.argmax(predictTe, axis=1) == teY) 
            print("score blend XGB")
            rXGBTe[j]=scoreTe
            print(scoreTe)


            predictVa=bmdl.predict_XGBproba(vaX)
            scoreVa=np.mean(np.argmax(predictVa, axis=1) == vaY) 
            print("score blend XGB val test")
            print(scoreVa)
            rXGBVa[j]=scoreVa

            print("blend nn")
            bmdl.fitNN(vaX,vaY,lambda1=0.0000005,lambda2=0.00001,teX=teX,teY=teY)


            predictTe=bmdl.predict_NNproba(teX)
            scoreTe=np.mean(np.argmax(predictTe, axis=1) == teY) 
            print("score blend nn")
            print(scoreTe)
            rNNTe[j]=scoreTe

            predictVa=bmdl.predict_NNproba(vaX)
            scoreVa=np.mean(np.argmax(predictVa, axis=1) == vaY) 
            print("score blend nn val set")
            print(scoreVa)
            rNNVa[j]=scoreVa

    print "rModTe"
    print rModTe
    print "rAveTe"
    print rAveTe
    print "rLogTe"
    print rLogTe
    print "rLogVa"   
    print rLogVa
    print "rLog2Te"
    print rLog2Te
    print "rLog2Va"
    print rLog2Va
    print "rXGBTe"
    print rXGBTe
    print "rXGBVa"
    print rXGBVa
    print "rNNTe"
    print rNNTe
    print "rNNVa"
    print rNNVa