"""
To run another instance of Perceptron class, I had to run in a new file
as doing in PerceptronPractice01.py led to a conflict

NOTE - conflict was only because I forgot to 'fit' the function in the loop!!!

THIS FILE: trying to organise a voting system on a limited number of runs,
a kind of crude early-stage ensemble method

ANALYSIS:
Very interesting!
> Seemingly best method for higher results is to increase the number of iterations. An issue there in some
cases will be that this can create over-fitting
> Good results if you increase the number of trials
> Also good results if more observations (rows in the dataset) to train the classifier on
So summary, this can be useful in situations with limited amount of observations (versus variables =
high dimensionality) and therefore danger of over-fitting

"""
# imports
import numpy as np
import matplotlib.pyplot as plt
import copy

# set random seed for lin_sep_data class
random_seed = 0
rgen = np.random.RandomState(random_seed)


""" 
Use the previous class I made to create the data. N.B. the Helper function

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
        self.data =np.column_stack((self.predictors, self.categories))

    def array(self):
        return self.data

    def help(self):
        print('predictors', self.predictors)
        print('targets', self.targets.shape)
        print('categories', self.categories)
        print('data', self.data)
        return ''


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


"""
For np random re-seeding, see: 
https://stackoverflow.com/questions/22994423/difference-between-np-random-seed-and-np-random-randomstate

For np array shape, e.g. (R, 1) or (R,), see:
https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r

"""
# create new dataset. N.B. You have to change X and y as you change the number of variables!!!
vote_data = lin_sep_data(rows=30, variables=4).array()
X = vote_data[:,:3]
y = vote_data[:,-1]
# print(vote_data)

""" 
N.B. the output vote_data.array, above, can't use the helper function.
Helper works as below
"""
test0 = lin_sep_data(rows=4, variables=3)
print("this prints all things inside the class!:\n", test0)
print(test0.help())


# fix number of trials
num_trials = 10
num_iter = 20

# create zeros array to add votes to
total_votes = np.zeros((X.shape[0],))
# print("total votes shape", total_votes.shape)
# print("total votes", total_votes)

# voting system with low number of iterations
for idx in range(num_trials):
    # n.b. to continously reseed the RandomState
    ppn = Perceptron(eta=0.1, n_iter=num_iter, random_state=idx)
    ppn.fit(X,y)
    temp = ppn.predict(X)
    temp1 = np.asarray(temp)
    total_votes += temp1

# this works nicely, below.  If > 0  means voted as 1, if < 0 voted as -1
print("total votes", total_votes)
# change to either 1 or -1
result = np.where(total_votes > 0, 1, -1)
print("result", result)
# actual result is y
print("actual", y)
# need to create an accuracy scoring system
comparison = ((result + y)/2)**2
print("comparison", comparison)
accuracy = np.sum(comparison)/X.shape[0]
print(accuracy)
print("accuracy after {0} trials with {1} iterations: {2:.2f}".format(num_trials, num_iter, accuracy))

# foo comment
