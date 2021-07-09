import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['label']=iris.target
df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]


data = np.array(df.iloc[:100, [0, 1, -1]])
#因为iloc[num_of_row_start : num_of_row_end, num_of_column_start : num_of_column_end]不包含num_of_end，所以需要 +1才能包含c行
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])

class Model:
    def __init__(self):
        self.w=np.ones(len(data[0])-1,dtype=np.float32)#变量为X1,X2;data[0]=[5.1,3.5,0]
        self.b=0
        self.l_rate=0.1

    def sign(self,w,x,b):
        y=np.dot(x,w)+b
        return y
    def fit(self,X_train,y_train):
        is_wrong=False
        while not is_wrong:
            wrong_count=0
            for d in range(len(X_train)):
                X=X_train[d]
                y=y_train[d]
                if y*self.sign(X,self.w,self.b)<=0:
                    self.w=self.w+self.l_rate*np.dot(y,X)
                    self.b = self.b + self.l_rate * y
                    wrong_count+=1
            if wrong_count==0:
                is_wrong=True
        return 'Perceptron Model!'


if __name__=='__main__':
    perceptron = Model()
    perceptron.fit(X, y)

    x_points = np.linspace(4, 7, 10)
    y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


