import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from loadfft import getData
import math

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.0001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    H={}
    h=[]
    H["h0"]=X
    nbLay=len(w)-1
    for i in range(1,nbLay+1):
        id="h"+`i`
        idp="h"+`i-1`
        H[id] = rectify(T.dot(H[idp], w[i-1]))
        H[id] = dropout(H[id], p_drop_hidden)
        h=np.append(h,H[id])
    py_x = softmax(T.dot(H[id], w[i]))
    return h, py_x

def loss(result,Y):
    l=len(Y)
    err=0
    for i in range(l):
        err+=-math.log(result[i,Y[i]])
    return err/l

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten().astype(int)
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def neuraln(trX,trY,params,nb_step=50,nbLay=5,nbNodes=150,p_drop_input=0.2,p_drop_hidden=0.4,size_output=4,lambda1=0.0000005,lambda2=0.0000):
    trY=one_hot(trY,size_output)
    print nb_step,nbLay,nbNodes,p_drop_input,p_drop_hidden
    check=nb_step/10
    
    X = T.fmatrix()
    Y = T.fmatrix()

    noise_h, noise_py_x = model(X,params, p_drop_input, p_drop_hidden)
    h, py_x = model(X, params,0,0)
    y_x = T.argmax(py_x, axis=1)
    
    L1=T.sum([T.sum(abs(params[i])) for i in range(len(params))])
    L2 = T.sum([T.sum((params[i])**2) for i in range(len(params))])  
    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))+lambda1*L1+lambda2*L2

    updates = RMSprop(cost, params, lr=0.001)
    
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
    predictProb=theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

    print(len(trX))
    i=0
    while i<nb_step:
        i+=1
        for start, end in zip(range(0, len(trX), 10), range(10, len(trX), 10)):
            cost = train(trX[start:end], trY[start:end])

        scoreTr=np.mean(np.argmax(trY, axis=1) == predict(trX))
        
        if i%(nb_step/20)==0:
            result=predictProb(trX)
            argY=np.argmax(trY, axis=1)
            logTr=loss(result,argY)
            print i,nbLay,scoreTr,logTr
            print cost,lambda1*L1.eval(),lambda2*L2.eval(),L1.eval(),L2.eval()
           
    return params

class NeuralNet():  
    def __init__(self,size_input,size_output=4,nb_step=200,nbLay=5,nbNodes=150,p_drop_input=0.2,p_drop_hidden=0.4):
        
        self.nb_step=nb_step
        self.nbLay=nbLay
        self.nbNodes=nbNodes
        self.p_drop_input=p_drop_input
        self.p_drop_hidden=p_drop_hidden
        params=[init_weights((size_input,nbNodes))]
        params.extend([init_weights((nbNodes,nbNodes)) for i in range(nbLay-1)])
        params.append(init_weights((nbNodes, size_output)))
        self.params=params
        self.size_input=size_input
        self.size_output=size_output
        
        
    def fit(self, trX,trY,lambda1=0.0000005,lambda2=0.00001,new=0):
        if not new:
            self.params= neuraln(trX,trY,self.params,self.nb_step,self.nbLay,self.nbNodes,self.p_drop_input,self.p_drop_hidden,self.size_output,lambda1,lambda2)  
        else:
            params=[init_weights((self.size_input,self.nbNodes))]
            params.extend([init_weights((self.nbNodes,self.nbNodes)) for i in range(self.nbLay-1)])
            params.append(init_weights((self.nbNodes, self.size_output)))
            self.params= neuraln(trX,trY,params,self.nb_step,self.nbLay,self.nbNodes,self.p_drop_input,self.p_drop_hidden,self.size_output,lambda1,lambda2)  
        return self
    
    def predict_proba(self,teX):
        X = T.fmatrix()
        h, py_x = model(X, self.params,0,0)
        predictProb=theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)
        return predictProb(teX)
          
        
if __name__ == '__main__':
    trX,trY,teX,teY=getData(oh=0)
    nn=NeuralNet(trX.shape[1])
    nn.fit(trX,trY)
    print np.mean(trY == np.argmax(nn.predict_proba(trX), axis=1))
    print np.mean(teY == np.argmax(nn.predict_proba(teX), axis=1))
