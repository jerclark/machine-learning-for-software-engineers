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


## Lesson 7  - Linear Discriminant Analysis

Similar to Logistic regression, although the target prediction is for multiple classes. Can also be used for
binary predictions, so it's worth a shot against logistic regression. Like the other regression algorithms, it
helps to remove highly correlated independent variables.

it calculates the mean and variance for each dependent variable, against each possible class.

Also, all of the independent variables must exhibit a normal (gaussian) distribution and have the same variance (mean of 0 standard deviation of 1)
This can be done by preparing the data beforehand.

Deeper dive here: http://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/


## Lesson 8 - Classification and Regression Trees

Decision Trees: A binary tree. Each node is an input/independent variable (x), with a split point on that variable
(assuming the value is numeric). The leaf nodes are predictions (y), made if the tree walk leads to that leaf (following
the splits).

Trees are fast to learn and fast to make predicitons, but they have high variance.


## Lesson 9 - Naive Bayes

Two types of probabilities in the model:
    the probability of each class
    The conditional probability of each class given a particular x value

Then the model is used to apply Bayes theroem to new samples.

Bayes Theorem - conditional probability:
P(A|B) = ( P(B|A)P(A) ) / P(B)
(Drug testing example here is a good one: https://en.wikipedia.org/wiki/Bayes%27_theorem)

When the independent variables are 'real-valued' its assumed they have a normal distribution (gaussian/bell)

It's called "naive" because it assumes that the x-values are independent (not correlated). This is a strong
assumption, and unlikely for real data - but this can be a cheap and effective algorithm.


## Lesson 10 - K Nearest Neighbors

Model is the entire data set.

New samples are compared to the most similar existing samples, and the
output variables are summarized to make a prediction.
For regression problems, perhaps the mean of the 'nearest' output variables and for classification
perhaps the 'mode' will be used.

The tricky part is to efficiently determine the 'nearest' neighbors...how is that
similarity defined? If all attributes (features/independent variables) are the same
scale, the euclidian distance (the simple difference between the exisiting sample and the new sample).

Can require lots of space, but only makes a computation (a learn) when necessary.

Closeness can 'break down' in high dimensions. When possible only use the attributes that are most relevant.

## Lesson 11 - Learning Vector Quantization

An artificial neural network algo. Unlike KNN which uses the entire training
data set (which you have to 'hold on' to), LVQ allows for choosing the
\# of training data samples to keep and it learns wheat they should look like.

The representation is a collection of 'codebook' vectors (what is that?) These are
summations of the dataset, learned over a set of iterations. Once learned, the codebook
vectors are used to make predictions like KNN. But i think the idea here is that you find
the closest 'codebook vector' and just use the target variable as the prediction.
(is this the same as CNN as described here?: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

Here's the wiki for LVQ: https://en.wikipedia.org/wiki/Learning_vector_quantization
They refer to "prototypes", which I think are the same as the 'codebook vectors' mentioned
in the python article. 

## Lesson 12 - Support Vector Machines

A 'hyperplane' is used to separate samples into classes. At simplest, the hyperplane
is a line in 2 dimensions, but can be in multiple dimensions as a complex curved plane. 

The distance between the plane and the nearest points is called the 'margin'. The optimal 
plane is the one that has the largest margin (that is, it best separates the classes.) The closest
datapoints are the only relevant points to define the hype plane - so they are called the "supports".

From: http://scikit-learn.org/stable/modules/svm.html

Advantages
* Effective in high dimensions
* Only use a subset of points in the decision function, so good on memory
* Versatile: Can use custom 'kernel' functions
* Can still be effective when features outnumber samples

Disadvantages:
* Can yeild poor results in features greatly outweight samples
* They don't provide probability estimates - must use an expensive five-fold cross validation

