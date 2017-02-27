# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import sklearn.svm as svm
from sklearn import tree
from sklearn import ensemble
import sklearn.linear_model as lm
from sklearn.metrics import r2_score
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
dtr = tree.DecisionTreeRegressor()
rfr = ensemble.RandomForestRegressor()
gbr = ensemble.GradientBoostingRegressor()
gbr_enhanced = ensemble.GradientBoostingRegressor(alpha=0.5, criterion='friedman_mse',
             init=None, learning_rate=0.1, loss='ls', max_depth=3,
             max_features=None, max_leaf_nodes=None,
             min_impurity_split=1e-07, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)
mse = 'neg_mean_squared_error'
r2 = 'r2'
models = [linregression, ridge, lasso, lars, omp, br, kn, svr, dtr, rfr, gbr, gbr_enhanced]
for model in models:
    mseResult = cross_val_score(model, X, Y, cv=kfold, scoring=mse)
    r2result = cross_val_score(model, X, Y, cv=kfold, scoring=r2)
    model = model.fit(X,Y)
    prediction = model.predict(X)
    print('%s: MSE: %.3f, R2: %.3f') % (model.__class__, mseResult.mean(), r2result.mean())
    print(r2_score(Y,prediction))
