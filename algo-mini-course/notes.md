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
