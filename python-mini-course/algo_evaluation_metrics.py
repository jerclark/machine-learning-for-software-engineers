# Cross Validation Classification LogLoss
import numpy
from pandas import read_csv
from pandas import read_table
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
dataframe = read_table(url, names=names, delimiter="\s+", na_values="?")
dataframe = dataframe[(~numpy.isnan(dataframe['horsepower']))]
array = dataframe.values
X = array[:,1:8]
Y = array[:,0]
model = LinearRegression()
model = model.fit(X, Y)
prediction = model.predict(X)
r2 = r2_score(Y, prediction)
print(r2)
