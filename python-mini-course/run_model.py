# Cross Validation Classification LogLoss
import pickle
import numpy
from pandas import read_table
from sklearn.metrics import r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
dataframe = read_table(url, names=names, delimiter="\s+", na_values="?")
dataframe = dataframe[(~numpy.isnan(dataframe['horsepower']))]
array = dataframe.values
X = array[201:,1:8]
Y = array[201:,0]
filename = "auto-displacment-model.sav"
model = pickle.load(open(filename, 'rb'))
prediction = model.predict(X)
r2 = r2_score(Y, prediction)
result = model.score(X, Y)

print("R2: %.3f, Score: %.3f") % (r2, result)

