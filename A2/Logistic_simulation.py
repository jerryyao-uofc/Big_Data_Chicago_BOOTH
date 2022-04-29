### Logistic simulation experiment

# Simulation experiment on logistic regression to visualize the decision boundary

import matplotlib.pyplot as plt
import math
plt.rc('font', size=14)
import seaborn as sns
sns.set(style='whitegrid', color_codes=True, rc={'figure.figsize':(15,8)}, font_scale=1.2)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

def Generate_data(data_input):
    '''
       Get simulated data in the proper form for further analysis.
       You don't need to modify this function.
    '''
    df = pd.DataFrame(data=data_input[0], columns=['Feature 1', 'Feature 2'])
    df['Label'] = data_input[1]
    return df

def mapFeature(X1, X2, degree):
    '''
       Generate the polynominal term of original feature. (The original feature is in 2 dimension)
       You don't need to modify this function.

       Input:  X1      : Feature 1, one dimension array (N * 1)
               X2      : Feature 2, one dimension array (N * 1)
               degree  : The max power of the polynominal term

       Output: res     : The polynomial term of original data.
    '''
    res = np.ones(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            res = np.column_stack((res, (X1 ** (i - j)) * (X2 ** j)))
    return res

def sigmoid(z):
    '''
       Sigmoid function
    '''
    return 1 / (1 + np.exp(-z))

def lossFunc(theta, X, y):
    '''
       Calculate the loss function for logistic regression
       Input:
       theta       : Coefficient for the hyper plane (decision boundary)
       X           : Feature data
       y           : Label data, (0,1) here

       Output:
       J           : Loss for logistic regression
    '''
    
    J = 0 
    for i in range(0, len(X)): 
        t = X[i]@theta
        J = J + math.log(1+math.exp(t)) - y[i]*(X[i]@theta.T)
    ##############################################################################
    ### TODO: Code up the loss function for logistic regression                ###
    ##############################################################################

    return J



def Get_gradient_numeric(theta, X, y):
    '''
       Calculate the gradient at one specific point, using numeric method.
       Input:
       theta       : Coefficient for the hyper plane (decision boundary)
       X           : Feature data
       y           : Label data, (0,1) here

       Output:
       gradient    : Calculated gradient using numeric method
    '''
    gradient = []

    # Step size h
    h = 1e-5

    for i in range(len(theta)):
        temp_pos = theta.reshape(len(theta), ).copy()
        temp_neg = theta.reshape(len(theta), ).copy()
        temp_pos[i] += h
        temp_neg[i] += -h
        gradient.append((lossFunc(temp_pos, X, y) - lossFunc(temp_neg, X, y)) / (2 * h) * X.shape[0])
    return gradient

def Get_gradient_formula(theta, X, y):
    '''
       Calculate the gradient at one specific point, using analytical formula.

       Input:
       theta       : Coefficient for the hyper plane (decision boundary)
       X           : Feature data
       y           : Label data, (0,1) here

       Output:
       gradient    : Calculated gradient using formula
    '''
    # gradient = 1/len(X) * X.T.dot(sigmoid(X.dot(theta)) - y)
    gradient = X.T.dot(sigmoid(X.dot(theta)) - y)

    # log likthlihoods 
    # gradient_1 = 0 
    # gradient_2 = 0
    # for i in range(0,len(X)):
    #     gradient_1  += (y[i] - sigmoid(X[i].dot(theta)))*X[i][0]
    #     gradient_2  += (y[i] - sigmoid(X[i].dot(theta)))*X[i][1]
    # gradient = np.array([[gradient_1],[gradient_2]])
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return gradient

def Vector_normalize(vector):
    '''
        Make the input vector has unit l2 norm
    '''
    return np.array(vector)/(np.array(vector)**2).sum() * np.sign(vector[-1])

def Gradient_check(gradient_1, gradient_2):
    '''
        Visualize the gradient for self check.
        You don't need to modify codes here.

        Input:
        gradient_1 : Calculated gradient based on numerical method
        gradient_2 : Calculated gradient based on analytical method

        Output:
        Scatter plot for two gradients
    '''

    # Make the vector have unit l2 norm
    gradient_1 = Vector_normalize(gradient_1)
    gradient_2 = Vector_normalize(gradient_2)
    print(gradient_1)
    print(gradient_2)
    return ((gradient_1-gradient_2)**2).sum()


def plotDecisionBoundary(theta, degree, x, y):
    '''
        Visualize the decision boundary.
        You don't need to modify codes here.

        Input:
        theta   : Coefficient for decision boundary
        degree  : The max power of the polynominal term
        x       : The min coordinate for visualization
        y       : The max coordinate for visualization

        Output:
        cs      : Contour plot of decision boundary
    '''

    u = np.linspace(x, y, 500)
    v = np.linspace(x, y, 500)
    U, V = np.meshgrid(u, v)
    # convert U, V to vectors for calculating additional features
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))

    X_poly = mapFeature(U, V, degree)
    Z = X_poly.dot(theta)

    # Reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))

    cs = plt.contour(U, V, Z, levels=[0], cmap="Greys_r")
    return cs

def Gradient_descent(X, y, stepsize, iteration):
    '''
        This procedure implements gradient descent algorithm

        Input:
        X        : Featured data
        y        : Label
        stepsize : Learning rate for gradient descent
        iteraton : Number of iterations

        Output:
        theta    : Optima after gradient descent algorithm

    '''
    # Initial theta value
    theta = np.zeros(X.shape[1])
  
    for i in range(iteration):
        theta = theta - stepsize* Get_gradient_formula(theta, X, y)



    return theta


def Generate_boundary(df, degree):
    '''
        This procedure implements logistic regression using gradient descent. 
        Steps :
            (1) Generate polynominal regressors as X
            (2) Run gradient descent to estimate logistic regression coefficients
            (3) Visualize the decision boundary

        Input:
        df      : Simulated dataset
        degree  : The max power of the polynominal term

    '''
    y = df.iloc[:,2];

    ##############################################################################
    ### TODO: Generate the polynominal regressors as X                         ###
    ##############################################################################
    
    X_raw = np.array(df[['Feature 1', 'Feature 2']])

    print(X_raw[1])
    poly = PolynomialFeatures(degree = degree)

    # X = (poly.fit_transform(X_raw))[:,1:]
    X = (poly.fit_transform(X_raw))
    print(X[1])
    # Calculate the decision boundary based on optimization

    theta = Gradient_descent(X, y, stepsize = 0.01, iteration = 100000)
    print(theta)
    # Visualization

    plt.figure(figsize=(12, 6))
    # Input data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature 1', y='Feature 2', hue='Label', data=df)
    plt.title('Input data')

    # Decision boundary
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Feature 1', y='Feature 2', hue='Label', data=df)
    plt.title('Decision boundary')
    
    X=df[['Feature 1', 'Feature 2']]


    x = min(X['Feature 1'].min(), X['Feature 2'].min()) -0.5
    y = max(X['Feature 1'].max(), X['Feature 2'].max()) +0.5

    plotDecisionBoundary(theta, degree, x, y)
        
