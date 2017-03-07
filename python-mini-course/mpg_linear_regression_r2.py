# Cross Validation Classification LogLoss
import numpy
from pandas import read_table
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import sklearn.linear_model as lm
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
dataframe = read_table(url, names=names, delimiter="\s+", na_values="?")
dataframe = dataframe[(~numpy.isnan(dataframe['horsepower']))]
array = dataframe.values
X = array[:,1:8]
Y = array[:,0]
linregression = lm.LinearRegression()
ridge = lm.Ridge()
lasso = lm.Lasso()
lars = lm.Lars()
omp = lm.OrthogonalMatchingPursuit()
br = lm.BayesianRidge()
model = ridge
model = model.fit(X, Y)
prediction = model.predict(X)
r2 = r2_score(Y, prediction)
mse = mean_squared_error(Y, prediction)
print("R2: %.3f, MSE: %.3f") % (r2, mse)
