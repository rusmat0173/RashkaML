"""
First, how numpy's random.RandomState works
Then lots of practice and look-see's at the functions in the Perceptron class (what are they, how do they work)
and lastly looking at how Perceptron works overall

IMPORTANT NOTE about ML algorithm, thank to the way this is executed in object orientated code:
The way object is written is very helpful, the single set of weighting (self.w_) is updated after looking at every observation
on every iterative run.  (self.w_ only keeps the last numbers used)

HOW DOES FIT WORK ON PREDICT? Since object keeps last self.w_, it automatically uses this on the test data

INTERESTING: found that going from 2 to 3 predictors really increases the number of iterations needed to go to zero errors.
nearly by one order of magnitude for a large dataset!  (Larger dataset needs more iterations.)

WHY IS THIS SO DIFFERENT TO IRIS DATA SET? (That dataset of 4 predictors converged
within 10 iterations.)  I think this is due to Iris having well separated categories.  In my data set there is a
sudden hard linear cut off

VALUES FOR TRAINING RATE? Are in interval [0:1). If these are high, you get to your Perceptron classification quickly
but can quickly oveshoot

N.B.s: use of import copy
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
import copy

# How np's randomRandomState works
"""
It doesn't give an output itself, you create an instance of a class 
which uses a particular pseudo-random number generator.
So you later have to call a method on that class, e.g. .normal or .randint
So you could use np.random.normal or np.random.RandomState.normal
It seems that the difference for RandomState is basically a different prn generator
and a slightly larger range of distributions to work from
"""
# example
random_seed = 0
rgen = np.random.RandomState(random_seed)
rand_array =rgen.normal(loc=0, scale=0.01, size=10)
print(rand_array)


# Some numpy linear algebra, can use a special linalg module
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
# dot product is basic matrix * matrix multiplication
c = np.dot(a,b)
print (c)
# find matrix inverse
d = np.array([[4,7],[2,6]])
e = np.linalg.inv(d)
# find matrix eigenvectors
f = np.array([[2,0],[0,2]])
g = np.linalg.eig(f)
print(g)


# practice '_' style
for _, _ in enumerate(range(3)):
    print(rgen.normal(loc=0, scale=0.01, size = 4))


# practice np.where; is quite amazing!
# https://stackoverflow.com/questions/34667282/numpy-where-detailed-step-by-step-explanation-examples
#  "tell me where in this array, entries satisfy a given condition"
h = np.array(rgen.random_integers(0, 9, size=(10)))
print(h)
i = np.where(h == 3)
print(i)
# more complex example: if h == 3, return 3, else 0
j = np.where(h == 3, 3, 0)
print(j)


# small play with int; True/False as integer - nice!
k = 7.3
l = int(k)
print(l)
# following returns True/False as integer; only works on single value
m = int(k != 0)
print(m)
# example from Perceptron class
errors = 0
update = [2.3, 5, -2]
for _ in update:
    errors += int(update != 0.0)
    print('errors: {}'.format(errors))


# small play with np.apply_along_axis and astype(int)
n = np.array([1.3, 2.4, 3.5, 4.0, 5.8, 6.6, 7.7, 8.95])
o = np.apply_along_axis(np.cumsum, arr=n, axis =0)
print(o)
p = o.astype(int)
print(p)


# create dataset for practicing on
""" this is with functions"""
np.random.seed(seed=200)
predictors = rgen.random_integers(0, 9, size=(20,3))
print(predictors[:3])

# create target variables
def create_targets(predictors, frac0, frac1):
    return np.add(predictors[:,0] * frac0, predictors[:,1] * frac1)

def create_targets2(predictors, linear_comb):
    cols = []
    for idx in range(len(linear_comb)):
        cols.append(predictors[:idx] * linear_comb[idx])
    return(cols)

# make target column as categorical
def make_categorical(targets):
    categories = []
    for target in targets:
        if target < 5:
            categories.append(-1)
        else:
            categories.append(1)
    return np.asarray(categories)

# numpy-esque method? (N.B. temp returns True/False vector)
def make_categorical2(targets, cutoff):
     temp = targets < cutoff
     return ((temp * 2) - 1) * -1

# merge to one dataset
def create_full_data(predictors, targets):
    return np.column_stack((predictors, targets))


targets = create_targets(predictors, 0.5, 0.6)
cat_targets = make_categorical2(targets, 5)
data = create_full_data(predictors, cat_targets)
print(data.size)

linear_comb = (0.5, 0.6, 0.2)
print(len(linear_comb))
targets2 = create_targets2(predictors, linear_comb)
print(targets2[:4])


# quick example classes exercise
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

    def function2(self):
        return "Another messsage"

myobjectx = MyClass()
print(myobjectx.variable)
myobjectx.function()
print(myobjectx.function2())


""" 
Now create data with a class, as easier to vary later on 
NOTE: use of helper function makes development life far easier as you can see 
what is happening as you go along

"""
class lin_sep_data:
    def __init__(self, rows=None, variables=None, proportions=(0.5, 0.5, 0.5), cutoff=5):
        self.predictors = rgen.random_integers(0, 9, size=(rows, variables))
        self.targets = np.zeros((rows,))

        # create targets as linear combinations of predictors
        for idx in range(len(proportions)):
            self.targets += self.predictors[:,idx] * proportions[idx]

        temp = self.targets < cutoff
        self.categories = ((temp * 2) - 1) * -1

        # merge full dataset
        self.data = np.column_stack((self.predictors, self.categories))

    def array(self):
        return self.data

    def help(self):
        print('predictors', self.predictors)
        print('targets', self.targets.shape)
        print('categories', self.categories)
        print('data', self.data)
        return ''


test0 = lin_sep_data(rows=4, variables=3)
print("this prints all things inside the class!:\n", test0)
print(test0.help())

""" full power and convenience here"""
test1 = lin_sep_data(rows=4, variables=3).array()
print(test1, '\n')


# create a Perceptron class as per Rashka, 2017.
# here I comment a lot to understand this.
class Perceptron(object):
    """
    Code for Perceptron given in Rashka 2017, plus lots of comments to understand how it works
    Also addition code to generate, e.g. list of the weightings at each iteration, to see what
    is going on under the hood
    """
    def __init__(self, eta=0.01, n_iter=25, random_state=4):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.list_of_weights =[]

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # below are the initial 'total' w_ value (w_[0]) and the weightings (w_[1:])
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+ X.shape[1])
        # create a list object to capture (to see how it works) how the weightings change at each iteration
        self.list_of_weights.append(copy.deepcopy(self.w_))
        self.errors_ = []

        for n in range(self.n_iter):
            errors = 0  # <= this is reinitialised every iteration
            for xi, target in zip(X,y):
                # clearly, this now iterates through every observation
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                temp_copy = copy.deepcopy(self.w_)
                self.list_of_weights.append(temp_copy)
                # below updates error if update !=0.  If update == 0, means correctly classified
                errors += int(update != 0.0)
            self.errors_.append(errors)

        # returns X, y
        return self

    def net_input(self, X):
        # this multiplies the X values (self.w_[1:])by the latest weightings using
        # vector dot product and adds back to previous self.w_ (self.w_[0])
        return (np.dot(X, self.w_[1:]) + self.w_[0])

    def predict(self, X):
        # says if net_input >=0, return 1, else return -1
        # shape of output is same as net_input(X)
        return np.where(self.net_input(X) >=0, 1, -1)

    # RA-made additional function
    def list_of_weights():
        """
        N.B. that despite using a fixed random seed, you don't get the same output
        even if you rerun the same command with same settings!
        """
        temp = self.list_of_weights[1:]
        return temp

    # RA-made additional function
    def key_weights(self):
        print('initial (random) weightings: {}'.format(self.list_of_weights[0][1:]))
        print('final weightings: {}'.format(self.list_of_weights[-1][1:]))
        return ''

    # outputing errors, here an additional function
        # could be simply done with print(object_name.errors_)
    def errors(self):
        error_list = []
        for idx, _ in enumerate(self.errors_):
            error_list.append((idx, _))
        return error_list


# test on the dataset
new_data = lin_sep_data(rows=100, variables=3).array()
X = new_data[:,:3]
y = new_data[:,3]

# X = predictors
# y = cat_targets
print(X.shape, y.shape)

ppn = Perceptron(eta=0.1, n_iter=50)
ppn.fit(X,y)

# print()
# print(ppn.key_weights())
# print(ppn.errors_)
# print(ppn.errors())

# plotting
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='x')
plt.xlabel('Epochs')
# An epoch is a measure of the number of times all of the training vectors
# are used once to update the weights.
plt.ylabel('Number of updates')
# plt.show()




