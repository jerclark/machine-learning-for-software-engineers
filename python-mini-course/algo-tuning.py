# Grid Search for Algorithm Tuning
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

# url = "https://goo.gl/vhm1eU"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = read_csv(url, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# model = Ridge()

url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
model = ensemble.GradientBoostingRegressor()


alphas = numpy.array([0.9999,0.1,0.01,0.001,0.0001,0.000000001])
param_grid = dict(alpha=alphas, random_state=[55555])
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_estimator_.alpha)