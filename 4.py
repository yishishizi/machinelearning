from csv import reader
from math import sqrt

def load_csv(filename):
    dataset=list()
    with open(filename,'r') as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if  not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def mean(values):
    return sum(values)/float(len(values))

def variance(values,mean):
    return sum([(x-mean)**2 for x in values])

def covariance(x,mean_x,y,mean_y):
    covar=0.0
    for i in range(len(x)):
        covar+=(x[i]-mean_x)*(y[i]-mean_y)
    return covar

def coefficients(dataset):
    x=[row[0] for row in dataset]
    y=[row[1] for row in dataset]
    mean_x=mean(x)
    mean_y=mean(y)
    covar=covariance(x,mean_x,y,mean_y)
    b1=covar/variance(x,mean_x)
    b0=mean_y-b1*mean_x
    return [b0,b1]


def  simple_linear_regression(train,test):
    predictions=list()
    b0,b1=coefficients(train)
    for row in test:
        yhat=b0+b1*row[0]
        predictions.append(yhat)
    return predictions

def rmse_metric(actual,predicted):
    sum_error=0
    for i in range(len(actual)):
        prediction_error=predicted[i]-actual[i]
        sum_error+=(prediction_error)**2
    mean_error=sum_error/float(len(actual))
    return sqrt(mean_error)

def evaluate_algorithm(dataset,algorithm):
    test_set=list()
    for row in dataset:
        row_copy=list(row)
        row_copy[-1]=None
        test_set.append(row_copy)#创建测试集
        print(test_set)
    prdicted=algorithm(dataset,test_set)
    print(prdicted)
    actual=[row[-1] for row in dataset]
    rmse=rmse_metric(actual,prdicted)
    return rmse



# Load dataset
#y=b0+b1*x
#filename = 'slr06.csv'
#dataset = load_csv(filename)
#for i in range(len(dataset[0])-1):
#	str_column_to_float(dataset, i)
# convert class column to integers
#str_column_to_int(dataset, len(dataset[0])-1)
#dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
filename = 'slr06.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))
rmse = evaluate_algorithm(dataset,simple_linear_regression)
print('RMSE: %.3f' % (rmse))


