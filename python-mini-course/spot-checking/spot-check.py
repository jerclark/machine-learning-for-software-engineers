# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import sklearn.svm as svm
import sklearn.linear_model as lm
url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
linregression = lm.LinearRegression()
ridge = lm.Ridge()
lasso = lm.Lasso()
lars = lm.Lars()
omp = lm.OrthogonalMatchingPursuit()
br = lm.BayesianRidge()
kn = KNeighborsRegressor()
svr = svm.SVR()
scoring = 'neg_mean_squared_error'
models = [linregression, ridge, lasso, lars, omp, br, kn, svr]
for model in models:
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    print('%s: %.3f') % (model.__class__, results.mean())