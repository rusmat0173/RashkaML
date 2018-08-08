"""
Aim here is to improve the linear separated data class and use it with Stochastic Gradient Descent
version of adaline (from Rashka, 2017).
After that, use Rashka's mlextend module to plot the variables in 2D and visualise the decision boundary.

ANALYSIS:
> Any form of batched gradient descent doesn't work so well on this data set. Depending on the coefficients
you can get misclassified data (e.g coefficients = [0.5,0.5]).
> Decision-making chart shows always some data on the borderline, and cost-graph doesn't reduce below a minimum.

N.B.:
> Using matplotlib this eay, you have to kill one chart then the other shows.


"""
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions

# better linear separated data class
class lin_sep_data2(object):
    """
    Will do in a Python list initially, due to the comments in following web link:
        https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy
    Aim is to have well-separated categories
    > Coefficients needs to be a list not a tuple, for it to work with one variable. (range(len(0.5)) not ok.)
    > Turned into an array at the end
    """
    def __init__(self, rows=50, coefficients=[0.5,0.5],  random_seed = 0):

        # initialise random seed of RandomState method

        self.coefficients = np.array(coefficients)
        self.data_ =[]
        self.count_ = 0
        rgen = np.random.RandomState(random_seed)
        low_cutoff = 0.33 * 9 * self.coefficients.size
        high_cutoff = 0.66 * 9 * self.coefficients.size

        while self.count_ < rows:
            self.row_ = []
            for _ in range(len(coefficients)):
                # ^ above is done from list to get length, even if length is one
                self.row_.append(rgen.randint(0, 9))
            scalar = np.dot(self.row_, self.coefficients)
            if scalar < low_cutoff:
                self.row_.append(-1)
                self.data_.append(self.row_)
            elif scalar < high_cutoff:
                self.row_.append(1)
                self.data_.append(self.row_)
            else:
                pass
            self.count_ += 1

        self.data_ = np.array(self.data_)

    def array(self):
        return self.data_


    def helper(self):
        print(self.data_)
        print(type(self.data_), self.data_.shape)


# Rashka's SGD version of Adaline
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# Rashka's other function for plotting decision regions in 2D, floem mlextned library that you have to install
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


# ANALYSE HERE
# create standardised dataset.  N.B. data is already shuffled i
data = lin_sep_data2(coefficients = [0.5, 0.5]).array()
np.random.shuffle(data)
X = data[:,:2]
X_scaled = preprocessing.scale(X)
y = data[:,-1]

# initiate class object and fit
ada = AdalineSGD(n_iter=25, eta=0.03, random_state=16)
ada.fit(X_scaled, y)


# plotting in matplotlib
plot_decision_regions(X_scaled, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel(' Col 0 [standardized]')
plt.ylabel('Col 1 [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('images/02_15_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
# plt.savefig('images/02_15_2.png', dpi=300)
plt.show()

# foo comment



