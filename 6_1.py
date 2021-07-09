from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,KFold,cross_val_score



def creat_data(filename):
    df=pd.read_csv(filename,header=0,sep=',')
    dataset=np.array(df.iloc[:,:])
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    return dataset[:,:-1],dataset[:,-1]



def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]#每次循环得到一行，每次制取第一行的第一个数，这条语句相当于取到了每个数据集的第一列
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

class logisticRression:
    def __init__(self,max_iter=1000,l_rate=0.01):
        self.max_iter=max_iter
        self.l_rate=l_rate

    def sigmod(self,x):
        return 1/(1+exp(-x))

    def data_matrix(self,X):
        m,n=X.shape
        A=np.ones((m,1))
        data_mat=np.c_[A,X]
        return data_mat

    def fit(self,X,y):
        data_mat=self.data_matrix(X)#将矩阵X转化成列表，并在第一列上添加1,m*n
        m,n=data_mat.shape
        self.weight=np.random.rand(n,1)
        for i in range(self.max_iter):
            for i in range(m):
               # print(np.dot(self.weight.T,data_mat[i].reshape(-1,1)))
                result=self.sigmod(np.dot(self.weight.T,data_mat[i].reshape(-1,1)))
                #print(self.weight)
                #print(data_mat[i].reshape(-1,1))
                error=y[i]-result
                self.weight+=self.l_rate*error*data_mat[i].reshape(-1,1)
        #print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
            #self.l_rate, self.max_iter))

    def score(self,X_test,Y_test):
        right=0
        X_test=self.data_matrix(X_test)
        for x,y in zip(X_test,Y_test):
            result=np.dot(self.weight.T,x.reshape(-1,1))
            if (result>0 and y==1) or (result<0 and y==0):
                right+=1
        return right/len(X_test)

np.random.seed(0)
filename='raw.csv'
X,Y=creat_data(filename)
kf = KFold(n_splits=10)
lc_clf=logisticRression()
for train_index, test_index in kf.split(X,Y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    #print(X_train, X_test)
    #print(Y_train, Y_test)
    lc_clf.fit(X_train, Y_train)
    print(lc_clf.score(X_test, Y_test))

X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.3)#----0.78
lc_clf.fit(X_train1,Y_train1)
print(lc_clf.score(X_test1,Y_test1))







