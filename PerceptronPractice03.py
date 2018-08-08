"""
abc
"""
import numpy as np

# initialise random seed
random_seed = 0
rgen = np.random.RandomState(random_seed)


""" Playing with creating alphabet lists """
# create a list or something, named by alphabet character
# technique 1 uses map, which is not a list
alphabet = map(chr, range(65, 91))
# print(type(alphabet))
# print(list(alphabet))
# print(list(alphabet))
# ^ above you see it gets emptied out! So try technique 2:
alphabet = list(map(chr, range(65, 91)))
# print(alphabet)
# print(alphabet)
# technique 3 is just to use a list:
# print(ord('A'))
alphabet = []
for letter in range(65, 91):
    alphabet.append(chr(letter))
# print(alphabet)
# print(alphabet[0])

# more testing
rgen = np.random.RandomState(random_seed * 0)
col_ = rgen.normal(loc=50.0, scale=16.7, size=100)
# print(col_)
# print(col_.shape)
col_ = rgen.uniform(low=0, high= 100, size=100)
# print(col_)
# print(col_.shape)

# for idx in range(1, 3):
#     print(idx)

# test = np.array([1,2,3,4])
# print(test.shape)
# test2 = np.array([4,3,2,1])
# print(test + test2)
# z = np.column_stack((test, test2))
# print(z)


# create regression_data class
class regression_data:
    """
    Class to create any kind of multiple regression array
    > Inputs: number of rows you want, and the 'true' coefficient of each variable (which gives # columns)
        and random seed
    > Outputs: Whole X,y array through .array(),
        last variable in array is the target variable, that is linear combination of the other columns
    > N.B.1:  needs at least 2 columns to work, else get into numpy array size/length issues
    > N.B.2:  for fun, alternates between columns created with 2 different distributions

    """

    def __init__(self, rows=10, coefficients=(0.5, 0.5, 0.5), random_seed=0):
        self.rows = rows
        self.coefficients = coefficients
        self.random_state = random_seed
        # create initial column to bind on.  (N.B. np.empty gives arbitrary very large values)
        rgen = np.random.RandomState(self.random_state -1)
        self.output_ = rgen.normal(loc=50.0, scale=16.7, size=self.rows)

        """ Note need to indent these functions!!! They have to be created before they are used """
        def normal(self, index):
            rgen = np.random.RandomState(self.random_state * index)
            self.col_ = rgen.normal(loc=50.0, scale=16.7, size=self.rows)
            return self.col_

        def uniform(self, index):
            rgen = np.random.RandomState(self.random_state * index)
            self.col_ = rgen.uniform(low=0, high=100, size=self.rows)
            return self.col_

        # we start variable columns from 1, as 0th one was created earlier
        # so we are alternating between 2 distributions, for fun
        for x in range(1, len(self.coefficients)):
            if x % 2 == 0:
                self.temp = normal(self, x)
            else:
                self.temp = uniform(self, x)
            self.output_ = np.column_stack((self.output_, self.temp))

        # create target variable
        """ 
        Usually it's X0 + B1X1 + B2X2 + ... + random error = Y
        To make easier, I will conceptually move random error to left-hand side ... = Y - random error
        But I will actually start with y as a narrowly spread random variable in the first place
        
        """
        self.y = rgen.normal(loc=0.0, scale=5.0, size=self.rows)

        for idx in range(len(self.coefficients)):
            temp = self.output_[:,idx] * coefficients[idx]
            self.y += temp
        self.output_ = np.column_stack((self.output_, self.y))

    def helper(self):
        print('rows', self.rows)
        print('coefficients', self.coefficients)
        print('random_state', self.random_state)
        print('col_', self.col_)
        print('y', self.y)
        print('output_', self.output_)
        return ''

    def array(self):
        return self.output_


regressor = regression_data(rows=5, coefficients=(0.5, 0.2, 0.6, 0.3), random_seed=5)
# print(regressor.helper())

# print(regressor.output_[:1], '\n')
print(regressor.array(), '\n')

# foo comment

