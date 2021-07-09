import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

def creta_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['lables']=iris.target
    df.columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data=np.array(df.iloc[:100,:])

    return data[:,:-1],data[:,-1]

X,y=creta_data()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

class NaiveBayes:
    def __init__(self):
        self.model=None

    @staticmethod
    def mean(X):
        return sum(X)/float(len(X))

    def stdev(self,X):
        avg=self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    def guassian_probability(self,x,mean,stdev):
         exponent=math.exp(-0.5*math.pow((x*mean)/stdev,2))
         return (1/(math.sqrt(2*math.pi)*stdev))*exponent

    def summarize(self,train_data):
         summaries=[(self.mean(i),self.stdev(i)) for i in zip(*train_data)]
         return summaries

    def fit(self, X, y):
        labels = list(set(y))  # set删除y中重复的元素，获取标签
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return "gaussianNB train done!"

    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.guassian_probability(input_data[i], mean, stdev)
        return probabilities

    def predict(self, X_test):
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))

if __name__=='__main__':
    model = NaiveBayes()
    model.fit(X_train, y_train)
    print(model.predict([4.4, 3.2, 1.3, 0.2]))





