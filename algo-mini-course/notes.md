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


## Lesson 4 -  Bias/Variance tradeoff

Bias: Assumptions made by algo to simplify. Can be easier to train and accurate, but less flexible.
High-bias - Linear algos
Low-bias - decision trees

Variance: Amount target function estimate will change if different training data used.
High-Var - knearest neighbors
Low-Var - Linear Discriminant Analysis

Goal to get low in both. Parameterization of algos does this.

Increase bias, decrease var
Increase var, decrease bias


