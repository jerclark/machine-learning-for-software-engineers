# Cross Validation Classification LogLoss
import numpy
from pandas import read_csv
from pandas import read_table
from sklearn.ensemble import GradientBoostingRegressor
import pickle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
dataframe = read_table(url, names=names, delimiter="\s+", na_values="?")
dataframe = dataframe[(~numpy.isnan(dataframe['horsepower']))]
array = dataframe.values
X = array[:,1:8]
Y = array[:,0]
model = GradientBoostingRegressor()
model = model.fit(X, Y)
filename = "auto-displacment-model.sav"
pickle.dump(model, open(filename, 'wb'))


prediction = model.predict(X)
r2 = r2_score(Y, prediction)
print(r2)
