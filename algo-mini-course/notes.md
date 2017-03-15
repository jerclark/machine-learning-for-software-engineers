## Lesson 1 - talking about data.

Output = f(input)
input: Rows contain "attributes" the columns
output: A predicted column value
variables or vectors (the inputs and outputs)


## Lesson 2 - The Principle That Underpins All Algorithms
Y = output prediction
X = input variables (vector of attributes)
Y = f(X), where f is the "model" or "predictive function"

Machine learning algorithms aim to learn what f is (often by training).


## Lesson 3 - Parametric and Nonparametric algos
Parametric: Make assumptions to try and simplify the function to be learned to a known form. The algorithms:

Select a form for the function.
Learn the coefficients for the function from the training data.
Examples - linear regression, logistic regression
Nonparametric: Make no assumptions about the form of the mapping function. As such, they can be more accurate but can also take more data and time to train.
Examples: SVC, neural networks, decision trees.


## Lesson 4 - Bias, variance and the trade off

Bias: Assumptions made by the model to make the training easier
Parametric functions generally have higher bias. Easier to train, but are less flexible. Can't perform well
on complex problems where the data won't subscribe to the assumptions.
High-bias: Linear Regression
Low-bias: Decision Trees


Variance: Amount estimate of the target function will change if different training data used.
The target function is the thing that's 'estimated' from the machine learning algorithm.
High-variance: KNearestNeighbors
Low-Variance: LinearDiscriminantAnalysis


Ideal is LOW bias and LOW variance. Parameterization of algorithms is a battle to balance bias and variance.

Increasing bias, decreases variance
Increasing variance, decreases bias


## Lesson 5 - Linear Regression

Linear Regression creates an equation describing a line that best 'fits' the relationship between inputs (X) and
outputs (Y) by finding weights for the input variables (Coefficients).

From the text:

For example:

y = B0 + B1 * x

We will predict y given the input x and the goal of the linear regression learning algorithm is to find the values for the coefficients B0 and B1.

Rule of thumb: Remove attributes that are similar (correlated) and remove noise from the data.

My Take:
Linear regression aims to devise a formula for a line, where the proper y-intercept and slope coefficient are
determined from the given data. One simple method is the 'least squares' method, where the square of the distance
of each point from the line is added up. (the sum of the squares of the vertical deviations). Smallest number wins. This can cause issues when there are outliers. Other types
of linear regressions are ridge regression and lasso regression. See here for good details: http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm

In ML, use the Y value of the line to predict for a given X.


## Lesson 6 - Logistic Regression

Similar to Linear regression - trying to find the coefficients to weight each input variable. However, the
target function is non-linear (the logistic function).

The logistic function will transform any output into a number between 0 and 1. It is used for binary classification
(two target classes). It works with binary classification by applying a rule to "snap" output values to either 0 or 1.
Like linear regression, removing highly correlated variables as well as variables that are unrelated to the output variable
improves the ability to estimate the target function.

Essentially, while the goal is to predict whether a sample is a 'case' or 'not a case', the output of the function
is a fraction representing the probablity that a sample is a 'case'.

https://en.wikipedia.org/wiki/Logistic_regression


## LESSON 8 - DECISION TEREES (CART)
Classification and regression trees.

Binary trees. each node is input variable (X) and the branch is based on a calculated split point. each leaf node is a prediction (Y). each x assumed to be numeric. for each inservation run the independent variables through the tree and see where end up.
split ppints are greedy,each one is optimized for the least cost to execute. the leaf nodes meet some "stopping criteria(like a min numbrer of samples in that class for the given path)
