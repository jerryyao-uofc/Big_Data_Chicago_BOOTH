### Logistic regression
import statsmodels.api as sm

# Perform the logistic regression

import pandas as pd

def Logistic_regression_function(X,y):
   '''
   perform the logistic regression ot predict default situation
   '''
   logit_model = sm.Logit(y,X)
   result = logit_model.fit()

 
   return result.pvalues.tolist(), result