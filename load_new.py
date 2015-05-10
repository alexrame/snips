__author__ = 'alexandrerame'

import numpy as np
import json
from pprint import pprint
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

random.seed(2)
np.random.seed(2)

def getData(prop=1./3, oh=1, pca=True, with_sfft=True, win_s=5, step=3):

    trX=[[]]
    trY=[]

    for y in range(4):
        file=getFile(y)
        with open(file) as data_file:
            data = json.load(data_file)

        size=len(data)
        print(size)

        for i in range(size):
            datX=np.array(data[i]['gravityBurst'])
            sizeDat=len(datX)

            accX=datX.flatten()[0:sizeDat:4][0:31].reshape((-1,1))
            accY=datX.flatten()[1:sizeDat:4][0:31].reshape((-1,1))
            accZ=datX.flatten()[2:sizeDat:4][0:31].reshape((-1,1))
            T=datX[3:sizeDat:4].flatten()[0:31]
            
            if len(accX)!=31 or len(accY)!=31 or len(accZ)!=31:
                continue
            # fourrier fenetre
            fft=np.array( [sfft(accX, win_s, step), sfft(accY, win_s, step), sfft(accZ, win_s, step)] ).reshape((1,-1))
            
            # main directions
            pca=PCA()
            #print accX.shape
            #print accY.shape
            #print accZ.shape
            
            temp=np.concatenate((accX, accY, accZ), axis=1)
            m=np.mean(temp, axis=0).reshape((1,-1))
            temp=temp-m
            temp=pca.fit_transform(temp)
            var=np.var(temp, axis=0).reshape((1,-1))
            for j in xrange(3):
                temp[:,j]=np.fft.fft(temp[:,j])
            
            main_sfft=sfft(temp[:,0], win_s, step).reshape((1,-1))
            
            # directions + var
            directions=pca.components_.reshape((1,-1))
            

            features=np.concatenate((fft, temp.reshape((1, -1)), main_sfft, m, var, directions), axis=1)
            features=features[0,:]
            
            # je laisse a voir
            if len(features)>=93:
                trX=np.append(trX,features)
                trY=np.append(trY,y)
            #else:
                #print(len(features))

                
    print(len(features))
    trX=np.reshape(trX,(-1,len(features)))
    l=len(trX)
    cut=l*prop
    
    arrayRandom=np.random.permutation(l)

    trX=trX[arrayRandom,:]
    trY=trY[arrayRandom]
    trX -= np.mean(trX, axis = 0)

    if oh:
        trY = one_hot(trY, 4)
    teX=trX[:cut,:]
    teY=trY[:cut]
    trX=trX[cut:,:]
    trY=trY[cut:]
    
    return trX,trY,teX,teY

def sfft(sig, win_s=15, step=3):
    start=0
    end=win_s
    s=np.array([])
    while end <= len(sig):
        
        s = np.append(s, np.real(np.fft.rfft( sig[start:end] )))
        s = np.append(s, np.real(np.fft.ifft( sig[start:end] )))
        
        start+=step
        end+=step
    return s

    
def getFile(y):
    if y==0:
        return "data/STILL.json"
    elif y==1:
        return "data/WALKING.json"
    elif y==2:
        return "data/RUNNING.json"
    else:
        return "data/BIKING.json"

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten().astype(int)
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def plotprint(vec,y):
    if y==0:
        plt.plot(vec,color='r')
    if y==1:
        plt.plot(vec,color='b')
    if y==2:
        plt.plot(vec,color='g')
    if y==3:
        plt.plot(vec,color='m')

if __name__ == '__main__':
    trX,trY,teX,teY=getData()
    print(teY)




