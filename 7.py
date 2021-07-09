from numpy import hstack
from numpy.random import normal
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

X1=normal(loc=20,scale=5,size=3000)
X2=normal(loc=40,scale=5,size=7000)
X=hstack((X1,X2))
X=X.reshape((len(X),1))

model=GaussianMixture(n_components=2,init_params='random')
model.fit(X)

yhat=model.predict(X)
print(len(yhat))
print('-------------------')
print(yhat[:100])
print('*******************')
print(yhat[-100:])