import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def creat_data(filename):
    df=pd.read_csv(filename,header=0,sep=',')
    data=np.array(df.iloc[:,:])
    return data[:,:-1],data[:,-1]

def minmax(dataset):
    minmax=list()
    for i in range(len(dataset[0])-1):
        row_values=[row[i] for row in dataset]#得到数据集中的每一列
        min_values=min(row_values)
        max_values=max(row_values)
        minmax.append([min_values,max_values])
    return minmax


def normal_dataset(minmax,dataset):
    for row in dataset:
        for i in range(len(row)):
            row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])

class MaxEnt:
    def __init__(self,trainDatalist,trainLabellist,testDatalist,testLabellist):
        self.trainDatalist=trainDatalist
        self.trainLabellist=trainLabellist
        self.testDatalist=testDatalist
        self.testLabellist=testLabellist
        self.featurenum=len(trainDatalist[0])

        self.N=len(trainDatalist)#训练集总长度
        self.n=0#训练集中(xi,y)对数量
        self.M=10000
        self.fixy=self.calc_fixy()#特征函数——所有(x,y)出现的次数
        self.w=[0]*self.n #权值w
        self.xy2idDict,self.id2xyDict=self.creatSearchDict()#(x, y)->id和id->(x, y)的搜索字典
        self.Ep_xy=self.calcEp_xy()#Ep_xy期望值

    def calcEpxy(self):
        #P_(x)*P(y|x)*logP(y|x)
        Epxy=[0]*self.n#P_(x)*P(y|x)*logP(y|x)，在python中, 如果用一个列表list1乘一个数字n 会得到一个新的列表list2, 这个列表的元素是list1的元素重复n次
        for i in range(self.N):
            Pwxy=[0]*2
            Pwxy[0]=self.calcPwy_x(self.trainDatalist[i],0)
            Pwxy[1]=self.calcPwy_x(self.trainDatalist[i],1)

            for feature in range(self.featurenum):
                for y in range(2):
                    if (self.trainDatalist[i][feature],y) in self.fixy[feature]:
                        id=self.xy2idDict[feature][(self.trainDatalist[i][feature],y)]
                        Epxy[id]+=(1/self.N)*Pwxy[y]
        return Epxy

    def clacEp_xy(self):
        EP_xy = [0 * self.n]
        for feature in range(self.featurenum):
            for(x,y) in self.fixy[feature]:
                id=self.xy2idDict[feature][(x,y)]
                EP_xy[id]=self.fixy[feature][(x,y)]/self.N

        return EP_xy

    def creatSearchDict(self):
        xy2idDict=[{} for i in range(self.featurenum)]
        id2xyDict={}

        index=0
        for feature in range(self.featurenum):
            for (x,y) in self.fixy[feature]:
                xy2idDict[feature][(x,y)]=index
                id2xyDict[index]=(x,y)
                index+=1

        return xy2idDict,id2xyDict

    def calc_fixy(self):
        fixyDict=[defaultdict(int) for i in range(self.featurenum)]
        for i in range(len(self.trainDatalist)):
            for j in range(self.featurenum):
                fixyDict[j][(self.trainDatalist[i][j], self.trainLabellist[i])] += 1

        for i in fixyDict:
            self.n+=len(i)
        return fixyDict

    def calcPwy_x(self,X,y):
        numerator=0
        Z=0

        for i in range(self.featurenum):
            if (X[i],y) in self.xy2idDict[i]:
                index=self.xy2idDict[i][(X[i],y)]
                numerator+=self.w[index]

        numerator=np.exp(numerator)
        Z=np.exp(Z)+numerator

        return numerator/Z

    def maxEntropyTrain(self,iter=500):
        for i in range(iter):
            Epxy=self.calcEpxy()
            sigmaList=[0]*self.n
            for j in range(self.n):
                sigmaList[j]=(1/self.M)*np.log(self.Ep_xy[j]/Epxy[j])

            self.w=[self.w[i]+sigmaList[i]for i in range(self.n)]


    def predict(self,X):
        result=[0]*2
        for i in range(2):
            result[i]=self.calcPwy_x(X,i)

        return result.index(max(result))

    def test(self):
        errorCnt=0

        for i in range(len(self.testDatalist)):
            result=self.predict(self.testDatalist[i])
            if result !=self.testLabellist[i]: errorCnt+=1

        return 1-errorCnt/len(self.testLabellist)


if __name__=='__main__':
    filename='raw.csv'
    X, Y = creat_data(filename)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    X_train=X_train.tolist()
    X_test=X_test.tolist()
    Y_train=Y_train.tolist()
    Y_test=Y_test.tolist()
    print(len(X_train))
    maxEnt=MaxEnt(X_train, X_test, Y_train, Y_test)
    maxEnt.maxEntropyTrain()