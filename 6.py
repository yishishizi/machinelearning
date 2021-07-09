from math import exp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def creat_data(filename):
    df=pd.read_csv(filename,header=0,sep=',')
    dataset=np.array(df.iloc[:,:])
    return dataset[:,:-1], dataset[:,-1]

def dataset_minmax(dataset):
	minmax=list()
	for i in range(len(dataset[0])):
		col_values=[row[i] for row in dataset]
		values_min=min(col_values)
		values_max=max(col_values)
		minmax.append([values_min,values_max])
	return minmax

def accuracy_metric(actual,predicted):
	correct=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			correct+=1
	return correct/float(len(actual))*100.0

def normalize_dataset(dataset,minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])


def predict(row,coefficients):
    yhat=coefficients[0]#数值b0
    for i in range(len(row)-1):
        yhat+=coefficients[i+1]*row[i]#coefficients是系数，row是变量的值
    return 1.0/(1.0+exp(-yhat))


def cofficients_sgd(train,l_rate,n_epoch):
	coef=[0.1 for i in range(len(train[0]))]
	print(coef)
	for epoch in range(n_epoch):
		sum_error=0
		for row in train:
			yhat=predict(row,coef)
			error=row[-1]-yhat
			sum_error+=error**2
			coef[0]=coef[0]+l_rate*error*yhat*(1.0-yhat)#b0(t+1) = b0(t) + learning_rate * (y(t) - yhat(t)) * yhat(t) * (1 - yhat(t))
			for i in range(len(row)-1):
				coef[i+1]=coef[i+1]+l_rate*error*yhat*(1.0-yhat)*row[i]#b1(t+1) = b1(t) + learning_rate * (y(t) - yhat(t)) * yhat(t) * (1 - yhat(t)) * x1(t)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		return coef

def logistic_algortihm(train,test,l_rate,n_epoch):
	prdicticons=list()
	coef=cofficients_sgd(train,l_rate,n_epoch)
	for row in test:
		yhat=predict(row,coef)
		prdicticons.append(round(yhat))
	return prdicticons




fileename='raw.csv'
X,Y=creat_data(fileename)
print(X)
dataset=np.c_[X,Y]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
train=np.c_[X_train,Y_train]
test=np.c_[X_test,Y_test]
l_rate=0.1
n_epoch=1000
prediction=logistic_algortihm(train,test,l_rate,n_epoch)
print(len(prediction),len(Y_test.tolist()))
score=accuracy_metric(Y_test.tolist(),prediction)
print(score)


