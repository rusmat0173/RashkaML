"""
Use of ADALINE improvement in Perceptron class.  ALSO A CLASSIFIER
Seems to have to significant differences:
i. Batch processing of predictors matrix in one go.
ii. Update the weightings based on previous weightings, and not do a +1/-1 classification at each step.

Instead of activation setting z(theta) either being +/-1, z(theta) is a batch process W(transpose) * X, where W and X are vectors
(of weights and variables respectively).  I.e. you compare continuously valued output against true class label.
I.e. you use activation function at each batch iteration, and finally use decision function.

The same z(theta) = +/-1, as per Perceptron, is used at final stage.  ('Decision function'.)

ALSO A COST FUNCTION!
This is Batch Gradient Descent, just like the one in Andrew Ng's ML MOOC video lectures, Rashka uses nearly identical-symbology maths
to Ng, so is a good reference.

DIFFERENCE WITH NG COST FUNCTION TECHNIQUE
> Doesn't add a column vector of all ones (X0 = 1, for all x) as Ng does.  This is because Ng
needs this for the intercept term in linear regression. (There is no intercept term in this classifier.)
> Also there is no division by 1/m.  Perhaps because a classifier???

WHAT'S NEW
> My first import of a module. See just below

N.B.
> Bizarre effect that because I imported a class from another file, that file's matplotlib's plt.show also shows up
(as not commented out)!

ANALYSIS OF ADALINE:
> This batch processing doesn't work as well as Perceptron, in fact plots of mean square errors show getting worse.
Possibly since my data is not well separated.
> Created much better separated version of same data type, made no difference.
> Unlikely to make a difference, as all data in same range,but will try NORMALISING DATA, as per book recommendations:
>> Will use an existing sklearn function for this. This come from TRANSFORM CLASS.
> ACTUALLY, it worked! Interesting that:
i. Normalising data worked much better.
ii. Still had a best sum of mean square errors of 9.  An misclassification gives an error of (e.g.) (-1 - 1) = 2.
Squared is MSE OF 4, so this means 2-and-a-bit misclassifications. (Doesn't quite make sense -  but no more than 3
misclassifications.)  This is worse than Perceptron.
iii. Plots very nicely show effect of too large and too small learning rates!
> Next file (PerceptronPractice05) try Stochastic Gradient Descent, which is Gradient Descent without the batching - so like Perceptron
with the weightings based on average previous of weightings, and a final +1/-1 classifier.

TO DO: I need to rewrite the linear separated dataset so it is more robust (also in next file)

"""
# imports
import numpy as np
import matplotlib.pyplot as plt

# first go at "importing modules", module's file needs to be on same path as this file
# in this case the regression_data class in PerceptronPractice03
from PerceptronPractice01 import lin_sep_data as lsd


# create Adaline class
class AdalineGD(object):
    """
    Code for Perceptron given in Rashka 2017, plus lots of comments to understand how it works
    Also addition code to generate, e.g. list of the weightings at each iteration, to see what
    is going on under the hood

    So what you do in fit function is to use partial derivative of cost function in update to
    self.w_ (batch processing of weights Gradient descent) but actual cost function
    (including squaring) for the cost itself that goes in its own list

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # below are the initial 'total' w_ value (w_[0]) and the weightings (w_[1:])
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # # create a list object to capture (to see how it works) how the weightings change at each iteration
        # self.list_of_weights.append(copy.deepcopy(self.w_))
        # new cost function list,
        # we are interested in cost not errors of self.errors = [], of last time
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            # Andrew Ng would reverse the terms in errors to be (output - y), but this is consistent with Perceptron
            errors = (y - output)
            # ^ above is a vector
            self.w_[1:] += self.eta * X.T.dot(errors)
            # ^ above is the batch operation. Unlike Ng. there is no division by 1/m,
            # where m is the number of observations of a variable Xj
            self.w_[0] += self.eta * errors.sum()
            # ^ above is the summation given in the formula
            cost = (errors**2).sum() / 2.0
            # ^ above is exact formula for cost function J
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """ calculate net input """
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, X):
        """ computes linear activation """
        return X

    def predict(self, X):
        """ return class label after each unit step """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)



# run this with matlab plotting
new_data = lsd(rows=50, variables=5).array()
X = new_data[:,:5]
y = new_data[:,5]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.3).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.3')

ada2 = AdalineGD(n_iter=5, eta=0.001).fit(X, y)
ax[1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.001')

# plt.savefig('images/02_11.png', dpi=300)
# plt.show()

# create separated data
sep_data = lsd(rows=100, variables=3, proportions=(0.5, 0.5, 0.5), cutoff=5).array()
print(sep_data[:5])
# some testing
test = sep_data[np.where(sep_data[:,3] == 1)]
print("test", test[:3], "\n", test.shape)
# create low and high data sets
low_data = sep_data[np.where(sep_data[:,0]* 0.5 + sep_data[:,1]* 0.5 + sep_data[:,1]* 0.5 < 5)]
high_data = sep_data[np.where(sep_data[:,0]* 0.5 + sep_data[:,1]* 0.5 + sep_data[:,1]* 0.5 > 10)]
print("LD", low_data.shape, low_data.size, type(low_data))
print("HD", high_data.shape, high_data.size, type(high_data))
sep_data = np.vstack((low_data, high_data))
np.random.shuffle(sep_data)
print(sep_data[:8])
# ^ all works

# now run new Adaline
X = sep_data[:,:3]
y = sep_data[:,3]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.3).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.3')

ada2 = AdalineGD(n_iter=5, eta=0.001).fit(X, y)
ax[1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.001')

# plt.savefig('images/02_11.png', dpi=300)
# plt.show()

# ^ That worked no better! Try standardising next (only X, y is a +1/-1 classifier)
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
print(X_scaled[:3,])
# ^ there's a warning given that we are ok with in this case.
#now just rerun the plotting, etc/. and see what happens to learning rate versus errors
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

ada2 = AdalineGD(n_iter=20, eta=0.3).fit(X_scaled, y)
ax[0].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.3')

ada2 = AdalineGD(n_iter=20, eta=0.001).fit(X_scaled, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.001')

ada2 = AdalineGD(n_iter=40, eta=0.0006).fit(X_scaled, y)
ax[2].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='x')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Sum-squared-error')
ax[2].set_title('Adaline - Learning rate 0.0003')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()

# foo comment