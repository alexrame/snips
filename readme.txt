#readme

load_new.py transforms the data into numpy arrays that we can use in our classification algorithms. We used Short-time-Fourier-transform (STFT) to create new, more relevant features. loadfft.py was the previous version that only uses fourier transfom and leads to a lower prediction accuracy.

xgb_search_params.py explores different sets of parameters for the gradient boosting trees algorithm including the data subsample, the features subsample, the learning rate and the maximum depth of trees. We keep in memory the different scores in a pickle. 

stacking.py builds a classifier called BlendedModel that takes as method the different stacking classifier that we will use, including logistic regression and neural network.

The neural network classifier is therefore encoded in neuralnet.py . You can select the number of layers, the number of nodes...

xgb_main.py takes the 10 best models that were saved in our pickle. We try many different kind of stacking classifiers that were defined in stacking.py. It returns the classification error for the different classifiers used during the stacking step.


