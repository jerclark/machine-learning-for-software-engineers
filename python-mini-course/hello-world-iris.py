import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as svm
from sklearn import tree
from sklearn import ensemble
import sklearn.linear_model as lm

# def r(data, colx, coly):
#     meanx = data.describe()[colx]["mean"]
#     meany = data.describe()[coly]["mean"]
#     stdx = data.describe()[colx]["std"]
#     stdy = data.describe()[coly]["std"]
#     n = data.shape[0]
#     s = 0
#     for ix, row in data.iterrows():
#         cx = ((row[colx] - meanx) / stdx)
#         cy = ((row[coly] - meany) / stdy)
#         s = s + (cx) * (cy)
#     r = s * (float(1) / (n - 1))
#     return r

#Read the data
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataframe = pandas.read_csv('./data/iris.data', header=None, index_col=None, names=names)

#Descriptive Stats
# print(dataframe.describe())
# print(dataframe.corr())

#Interesting plots
# dataframe.hist()
# dataframe.plot(type="box")
# pandas.scatter_matrix(dataframe)

#Cross Validation
array = dataframe.values
X = array[:,0:4]
Y = array[:,4]
kfold = KFold(n_splits=10, random_state=7)
# linregression = lm.LinearRegression()
# ridge = lm.Ridge()
# lasso = lm.Lasso()
# lars = lm.Lars()
# omp = lm.OrthogonalMatchingPursuit()
# br = lm.BayesianRidge()
kn = KNeighborsClassifier()
svc = svm.SVC()
lsvc = svm.LinearSVC()
dt = tree.DecisionTreeClassifier()
rf = ensemble.RandomForestClassifier()
gb = ensemble.GradientBoostingClassifier()
bag = ensemble.BaggingClassifier(svc)
models = [kn, svc, lsvc, dt, rf, gb, bag]
for model in models:
    result = cross_val_score(model, X, Y, cv=kfold)
    print('%s: Score: %.3f') % (model.__class__, result.mean())



