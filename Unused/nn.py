import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from loadfft import getData

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

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
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

trX,trY,teX,teY=getData()

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


def neuraln(nb_step=1000,nbLay=5,nbNodes=100,p_drop_input=0.1,p_drop_hidden=0.1):
    print nb_step,nbLay,nbNodes,p_drop_input,p_drop_hidden
    check=nb_step/10
    
    X = T.fmatrix()
    Y = T.fmatrix()

    params=[init_weights((93,nbNodes))]
    params.extend([init_weights((nbNodes,nbNodes)) for i in range(nbLay-1)])
    params.append(init_weights((nbNodes, 4)))
    
    noise_h, noise_py_x = model(X,params, 0.2, 0.5)
    h, py_x = model(X, params, p_drop_input, p_drop_hidden)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    updates = RMSprop(cost, params, lr=0.001)

    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
    
    maxscore=0
    checkvalue=0
    print(len(trX))
    i=-1
    while i<nb_step:
        i+=1
        for start, end in zip(range(0, len(trX), 10), range(10, len(trX), 10)):
            cost = train(trX[start:end], trY[start:end])
        scoreTe=np.mean(np.argmax(trY, axis=1) == predict(trX))
        scoreTr=np.mean(np.argmax(teY, axis=1) == predict(teX))
        if i%1==0:
            print i,nbLay,scoreTr,scoreTe
        if scoreTe>maxscore: 
            maxscore=scoreTe
            imax=i
        if i%check==0:
            if checkvalue<scoreTr:
                checkvalue=scoreTr
            else:
                i=nb_step
        if i==(nb_step-1):
            print("predict trX")
            print(predict(trX))
            print("predict teX")
            print(predict(teX))
            print("teY")
            print(np.argmax(teY, axis=1))
    
    return imax,nbLay,maxscore

if __name__ == '__main__':
    nbLay,trScore,teScore,score, imax=neuraln()
    #nbLay2,trScore2,teScore2,score2, imax2=neuraln(nbLay=10)
    print nbLay,trScore,teScore,score, imax
    #print nbLay2,trScore2,teScore2,score2, imax2