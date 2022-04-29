import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True,rc={'figure.figsize':(12,8)})
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")



def Normalize(df):
    '''
       Use this function to normalize the input DataFrame.
       Each column should have zero mean and unit variance after the normalization.
       You don't need to change any codes here.

       Input:
       df : The raw DataFrame

       Output:
       df_stdize : The DataFrame after column normalization.
    '''
    scaler = StandardScaler()
    df_stdize = scaler.fit_transform(df)
    df_stdize = pd.DataFrame(df_stdize)
    df_stdize.index = df.index
    df_stdize.columns = df.columns

    return df_stdize


def Portfolio_construction(df, alp=0.06):
    '''
       Use this function to construct your portfolio.

       Input:
       df : The DataFrame used for further analysis, can be price, return

       Note:
       You can add your own input variables in this function.

       Output:
       portfolio_weight : Array, the assigned weight for each stock to track the S&P 500 index.
       y_ture           : Array, Real index OOS return
       y_predict        : Array, Your predicted OOS return

    '''

    # Normalize the input DataFrame
    df = Normalize(df)

    y = df.SPX
    X = df.iloc[:, :-1]

    # Train test split
    n_sample = X.shape[0]
    X_train = X.iloc[:int(0.6 * n_sample), :]
    y_train = y[:int(0.6 * n_sample)]
    X_test = X.iloc[int(0.6 * n_sample):, :]
    y_test = y[int(0.6 * n_sample):]


    y_true = y_test
    lasso_reg = Lasso(max_iter=10000)
    lasso_model = lasso_reg.set_params(alpha=alp).fit(X_train, y_train)
    y_predict = lasso_model.predict(X_test)
    
    # need some engineering to make y_predict into the same format as y_true
    processed_predict = pd.concat([pd.Series(y_test.index.tolist()),  pd.Series(y_predict.tolist())], axis=1)
    
    portfolio_weight = lasso_model.coef_
    return portfolio_weight, y_true, processed_predict.set_index(0).squeeze()


def Portfolio_visualize(y_true, y_predict):
    '''
       Use this function to visualize your portfolio.
       You don't need to modify any codes here.

       Input:
       y_ture           : Array, Real index OOS return
       y_predict        : Array, Your predicted OOS return
    '''

    df = (pd.concat([y_true, y_predict], axis=1)).dropna()
    df.columns = ['Index return', 'Portfolio return']

    # Plot the cumulative return
    np.cumsum(df).plot()


def Portfolio_rebalance(df, window, alp = 0.06):

    '''
       Use this function to rebalance your portfolio.

       Input:
       df     : The DataFrame used for further analysis, return of SP500 here
       window : The length of time period for rebalancing, set window = 60 here

       Note:
       You can add your own input variables in this function.

       Output:
       Portfolio_weight      : DataFrame, the assigned weight for each stock to track the S&P 500 index.
       Portfolio_performance : DataFrame, the OOS performance for your tracking strategies.

    '''

    # Initialization
    Portfolio_weight = pd.DataFrame(index=df.index, columns=df.columns[:-1])
    Portfolio_performance = pd.DataFrame(index=df.index, columns=['Predicted Value'])
  
    # print(Portfolio_performance)
    # # Standize the original data
    df = Normalize(df)

    y = df.SPX
    X = df.iloc[:, :-1]

    for period in range(int(df.shape[0] / window)):

        # Get the training period and OOS period
        X_train = X.iloc[window * period:window * (period + 1), :]
        y_train = y[window * period:window * (period + 1)]
        X_test = X.iloc[window * (period + 1):window * (period + 2), :]
        y_test = y[window * (period + 1):window * (period + 2)]

        # print("X_train shape", X_train.shape)
        # print("y_train shape", y_train.shape)
        # print("X_test shape", X_test.shape)
        # print("y_test shape", y_test.shape)
        ##############################################################################
        ### TODO: Design your portfolio rebalancing method here                    ###
        ##############################################################################
        # Initialization
        
        lasso_reg = Lasso(max_iter=10000)
        lasso_model = lasso_reg.set_params(alpha=alp).fit(X_train, y_train)
        y_predict = lasso_model.predict(X_test)
        y_predict = y_predict.reshape(y_predict.shape[0],1)
        #print(y_predict)
        
        portfolio_weight = lasso_model.coef_
       
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        # Store the portfolio weight and OOS performance
        Portfolio_performance.iloc[window * (period + 1):window * (period + 2)] = y_predict
        Portfolio_weight.iloc[window * (period + 1), :] = portfolio_weight

    return Portfolio_weight, Portfolio_performance





