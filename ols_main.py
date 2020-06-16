# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:19:13 2020

@author: Professional
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
import random
from sklearn import datasets
import itertools
import math
import time
from scipy.stats import f,t
import os

ROOT =  os.path.dirname(os.path.realpath(__file__))
FILENAME = 'prediction.xlsx'

def main():
    """ols_main.py's main function. Does the following:
        
        1. input independent and dependent variable data is split into 
	       "training" and "prediction" sets. 
    	2. Within the training set, k-fold crossvalidation is used to 
    	   generate an Akaike Information Criteria (AIC) value for each 1-p 
    	   combinations of independent variables. 
    	3. The model with the lowest AIC is selected and fit to the entire 
    	   training set.
    	4. The newly fit champion model is used to predict values using the 
    	   independent variables of the "prediction" set, withheld in (1).
    	5. The predicted values are compared against the actual values of the
    	   dependent variable withheld in (1), and common regression 
    	   statistics are generated and stored in an Excel File, named 
    	   "prediction.xlsx"
    
    Arguments:
        
        None
                    
    Returns:
        
        None
        
    Raises Error:
        
        None
        
    """
    # Load "toy" dataset
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    data = shuffleData(X,y)
    
    # Split into training and prediction sets
    thresh = math.floor(data.shape[0]*0.8)
    trainX = data[:thresh,1:]
    trainY = data[:thresh,:1]
    outX = data[thresh:,1:]
    outY = data[thresh:,:1]
    
    # Run OLS Procedure
    start = time.time()
    print("Finding best model...")
    champion = findBestModel(trainX,trainY,boston.feature_names)
    print("Fitting champion model...")
    champion = fitBestModel(champion,trainX,trainY,boston.feature_names)
    ex_time = time.time() - start
    if ex_time > 60:
        print("Champion model fit complete (%s minutes)." % round((ex_time/60),2))
    else:
        print("Champion model fit complete (%s seconds)" % round(ex_time,2))
    
    # Associate regressor position with regressor names.
    indexes = []
    for name in champion.feature_names:
        indexes.append(boston.feature_names.tolist().index(name))
    outX = outX[:,indexes]
    
    # Predict Y value using champion model, report to Excel.
    champion.predict(outX,outY)
    champion.reportPrediction(outY)

def bubbleSort(sort_list): 
    """Bubble Sort
    
    A simple bubble sort method used to determine the champion model. Sorts by
    AIC in ascending fashion. Time complexity: O(n^2)
    
    Arguments:
        
        sort_list (List) - An unsorted list of OLSRegModel objects to be sorted 
        in place by AIC.
                    
    Returns:
        
        sort_list (List) - A sorted list of OLSRegModel objects sorted by AIC.
        
    Raises Error:
        
        None
        
    """
    l = len(sort_list) 
    for i in range(0, l): 
        for j in range(0, l-i-1): 
            if (sort_list[j].crossValStats.aic > sort_list[j + 1].crossValStats.aic): 
                tempo = sort_list[j] 
                sort_list[j]= sort_list[j + 1] 
                sort_list[j + 1]= tempo 
    return sort_list 

def findBestModel(X,y,feature_names = None):
    """Investigates all possible models given p regressors to crown a champion
    model.
    
    Generates an array of all 1-p combinations of regressors, each of which
    are crossvalidated to generate an AIC value for model comparison.
    
    Arguments:
        
        X (Numpy Array) - An n-by-p array of floats depicting each observation
        (row) of each regressor (independent variable)(column).
        
        y (Numpy Array) - An n-by-1 array of floats depicting each ovsercation
        (row) of the response (dependent) variable.
                          
        feature_names (Numpy Array) - Default = None - An n-by-1 array of 
        strings representing the name of each regressor. If left as default,
        the function will assign names such as "X1."
                    
    Returns:
        
        models[0] (OLSRegModel) - The champion model object.
        
    Raises Error:
        
        None
        
    """
    number = list(range(X.shape[1]))
    allcombos = []
    for i in range(1,X.shape[1]):
        results = itertools.combinations(number,i)
        combos = np.array(list(results))
        allcombos.append(combos)
    
    models = []
    for i in range(len(allcombos)):
        combos = allcombos[i]
        for j in range(combos.shape[0]):
            indVars = combos[j,:]
            data = X[:,indVars]
            mdl = OLSRegModel(feature_names[allcombos[i][j,:]].tolist())
            mdl.crossValidate(data,y)
            models.append(mdl)
            
    models = bubbleSort(models)
    
    return models[0]

def fitBestModel(model,X,y,full_feature_names = None):
    """Trains the champion model on full set of "training" data.
    
    Generates "beta" for the champion model based on full set of training
    data, to be used in prediction.
    
    Arguments:
        
        X (Numpy Array) - An n-by-p array of floats depicting each observation
        (row) of each regressor (independent variable)(column).
        
        y (Numpy Array) - An n-by-1 array of floats depicting each observation
        (row) of the response (dependent) variable.
                          
        full_feature_names (Numpy Array) - Default = None - An n-by-1 array of 
        strings representing the name of each regressor. If left as default,
        the function will assign names 'X0' through 'Xp' as names.
                    
    Returns:
        
        mdl (OLSRegModel) - The champion model object, with an updated
        "mdl.beta" attribute to be used in prediction.
        
    Raises Error:
        
        None
        
    """
    if model.feature_names is None:
        model.feature_names = ['x'+str(i) for i in X.shape[1]]
    if full_feature_names is None:
        full_feature_names = ['x'+str(i) for i in X.shape[1]]
        
    indexes = []
    for name in model.feature_names:
        indexes.append(full_feature_names.tolist().index(name))
    indexes = np.array(indexes)
    X = X[:,indexes]
    mdl = OLSRegModel(model.feature_names)
    mdl.train(X,y)
    return mdl

def shuffleData(X,y):
    """A helper function to shuffle regression data.
    
    Shuffles a set of independent and dependent variables, maintaining row
    associations.
    
    Arguments:
        
        X (Numpy Array) - An n-by-p array of floats depicting each observation
        (row) of each regressor (independent variable)(column).
        
        y (Numpy Array) - An n-by-1 array of floats depicting each observation
        (row) of the response (dependent) variable.
                    
    Returns:
        
        data (Numpy Array) - An n-by-p+1 array of floats depicting each
        observation (row) of the response (dependent) variable and regressors
        (independent variables)(columns).
        
    Raises Error:
        
        None
        
    """
    data = np.hstack([np.column_stack(y).T,X])
    np.random.shuffle(data)
    return data

class OLSRegModel:
    """An OLS Regression Model Object is used to store methods and data
    necessary to conduct OLS Regression analysis."""
    
    def __init__(self, feature_names = None):
        """Initializes the OLSRegModel object, storing the given feature names
        and creating an empty 'predicted' attribute. The rest of the classes
        attributes are added within the classes functions."""
        self.feature_names = feature_names
        self.predicted = None
        
    def checkIntercept(self,X):
        """A helper function to check if X has an intercept column.
    
        This regression algorithm assumes an intercept for its OLS procedure
        
        Arguments:
            
            X (Numpy Array) - An n-by-p array of floats depicting each observation
            (row) of each regressor (independent variable)(column).
            
            y (Numpy Array) - An n-by-1 array of floats depicting each observation
            (row) of the response (dependent) variable.
                        
        Returns:
            
            data (Numpy Array) - An n-by-p+1 array of floats depicting each
            observation (row) of the response (dependent) variable and regressors
            (independent variables)(columns).
            
        Raises Error:
            
            None
            
        """
        if not (X[:,0] == np.ones([X.shape[0],1])).all():
            X = np.concatenate([np.ones([X.shape[0],1]),X],axis = 1)
        return X
    
    def crossValidate(self,X,y,k = 5,feature_names = None):
        """Conducts K-fold Cross validation on the given data.
        
        Uses k-fold cross validation to calculate regression statistics for an 
        OLSRegModel object fit to the given data. The function creates the
        OLSRegModel's crossValStats object, and assigns the results of the 
        crossvalidation to the object.
        
        Arguments:
            
            X (Numpy Array) - An n-by-p array of floats depicting each observation
            (row) of each regressor (independent variable)(column).
            
            y (Numpy Array) - An n-by-1 array of floats depicting each observation
            (row) of the response (dependent) variable.
            
            k (int) - An integer determining the number of "folds" in the cross
            validation. A fold is a subset of the data that is subsequently 
            used as both a training set and a validation set.
            
            feature_names (Numpy Array) - Default = None - An n-by-1 array of 
            strings representing the name of each regressor. If left as default,
            the function will assign names such as "X1."
                        
        Returns:
            
            None
            
        Raises Error:
            
            None
            
        """
        self.crossValStats = self.RegStats(proc = 'cval')
        
        data = shuffleData(X,y)
        allfolds = np.array_split(data,k)
        random.shuffle(allfolds)
        
        # Regression Statistics
        r2 = []
        mr = []
        r2_adj = []
        se = []
        aic = []
        bic = []
        rmse = []
        
        for i in range(k):
            folds = allfolds.copy()
            outfold = folds.pop(i)
            infolds = folds
            indata = np.concatenate(infolds,axis = 0)
            
            # Define training and validation data
            Xin = indata[:,1:]
            Yin = indata[:,:1]
            Xout = outfold[:,1:]
            Yout = outfold[:,:1]
            
            # Train new model, validate
            self.train(Xin,Yin)
            self.validate(Xout,Yout)
        
            # Store Regression Statistics
            r2.append(self.regStatsVal.r2)
            mr.append(self.regStatsVal.r2)
            r2_adj.append(self.regStatsVal.r2_adj)
            se.append(self.regStatsVal.se)
            aic.append(self.regStatsVal.aic)
            bic.append(self.regStatsVal.bic)
            rmse.append(self.regStatsVal.rmse)
        
        # Average Statistics
        self.crossValStats.r2 = sum(r2)/k
        self.crossValStats.mr = sum(mr)/k
        self.crossValStats.r2_adj = sum(r2_adj)/k
        self.crossValStats.se = sum(se)/k
        self.crossValStats.aic = sum(aic)/k
        self.crossValStats.bic = sum(bic)/k
        self.crossValStats.rmse = sum(rmse)/k
    
    def predict(self,X,actual):
        """Predict response variable given an array of regressors.

        Arguments:
            
            X (Numpy Array) - An n-by-p array of floats depicting each observation
            (row) of each regressor (independent variable)(column).
            
            actual (Numpy Array) - An n-by-1 array of floats depicting each 
            observation (row) of the response (dependent) variable's true 
            value.
                        
        Returns:
            
            None
            
        Raises Error:
            
            None
            
        """
        X = self.checkIntercept(X)
        self.predicted = np.dot(X,self.beta)
        
        self.regStatsPred = self.RegStats(X,actual,self.beta,'pred')
        
    def reportPrediction(self,actual):
        """Export numerous regression statistics to Excel
        
        Neatly exports regression fit statistics, residual values, standardized
        residual values, ANOVA, Predicted vs Actual scatterplot, and Predicted 
        vs Standardized Residual scatterplot to be used for statistical 
        analysis of model fit.

        Arguments:
            
            actual (Numpy Array) - An n-by-1 array of floats depicting each 
            observation (row) of the response (dependent) variable's true 
            value.
                        
        Returns:
            
            None
            
        Raises Error:
            
            None
            
        """
        if self.predicted is None:
            raise ValueError("No prediction data available" + \
                             "(model's prediction attribute is null).")
        
        if not os.path.exists(ROOT + '/output/'):
            os.makedirs(ROOT + '/output/')
        
        writer = pd.ExcelWriter(ROOT + '/output/' + FILENAME, 
                                engine='xlsxwriter')
        sheet_name = 'Sheet1'
        
        # Actual vs. Predicted Data
        data = pd.concat([pd.DataFrame(self.predicted, 
                                       columns = ['Predicted']),
                          pd.DataFrame(actual,columns = ['Actual'])],
                         axis = 1)
        data['Residuals'] = (data['Predicted']-data['Actual']) 
        data['s.Residuals'] = (data['Predicted']-data['Actual']) / data['Actual']**(1/2)
        data.to_excel(writer, sheet_name=sheet_name, index = False)
        
        # Prediction Regression Statistics
        regstats = pd.DataFrame(columns = ['Statistic','Value'])
        regstats = regstats.append({'Statistic':'R2','Value':self.regStatsPred.r2},
                                   ignore_index = True)
        regstats = regstats.append({'Statistic':'MR','Value':self.regStatsPred.mr},
                                   ignore_index = True)
        regstats = regstats.append({'Statistic':'Adjusted R2','Value':self.regStatsPred.r2_adj},
                                   ignore_index = True)
        regstats = regstats.append({'Statistic':'SE','Value':self.regStatsPred.se},
                                   ignore_index = True)
        regstats.to_excel(writer, sheet_name = sheet_name, startcol = 5, index = False)
        
        # ANOVA
        anova1 = pd.DataFrame(columns = [' ','df','SS','MS','F','Significance F'])
        anova1 = anova1.append({' ': 'Regression',
                                'df':self.regStatsPred.df_reg,
                                'SS':self.regStatsPred.sse,
                                'MS':self.regStatsPred.mse_reg,
                                'F':self.regStatsPred.fstat,
                                'Significance F':self.regStatsPred.fsig},
                               ignore_index = True)
        anova1 = anova1.append({' ': 'Residual',
                                'df':self.regStatsPred.df_res,
                                'SS':self.regStatsPred.ssr,
                                'MS':self.regStatsPred.mse_res,
                                'F': None,
                                'Significance F':None}, 
                               ignore_index = True)
        anova1 = anova1.append({' ': 'Total',
                                'df':self.regStatsPred.df_tot,
                                'SS':self.regStatsPred.sst,
                                'MS':None,
                                'F': None,
                                'Significance F':None}, 
                               ignore_index = True)
        anova1.to_excel(writer,sheet_name = sheet_name,startrow = 6,startcol = 5, index = False)
        
        anova2 = pd.concat([pd.DataFrame(np.insert(self.feature_names,0,'Int.'),columns = [' ']),
                            pd.DataFrame(self.beta,columns = ['Coefficients']),
                            pd.DataFrame(self.regStatsPred.se_coeff,columns = ['Standard Error']),
                            pd.DataFrame(self.regStatsPred.t_stat,columns = ['t Stat']),
                            pd.DataFrame(self.regStatsPred.p_vals,columns = ['P-value']),
                            pd.DataFrame(self.regStatsPred.lower,columns = ['Lower 95%']),
                            pd.DataFrame(self.regStatsPred.upper,columns = ['Upper 95%'])],
                           axis = 1)
        anova2.to_excel(writer,sheet_name = sheet_name,startrow = 11,startcol = 5, index = False)
        
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Add scatterplots
        chart = workbook.add_chart({'type': 'scatter'})
        chart.add_series({'name': 'Actual vs. Predicted',
                          'categories': sheet_name + '!$A$2:$A$' + str(data.shape[0]),
                          'values': sheet_name + '!$B$2:$B$' + str(data.shape[0]),
                          'trendline': {'type':'linear'}})
        chart.set_x_axis({'name': 'Predicted Value'})
        chart.set_y_axis({'name': 'Actual Value',
                          'major_gridlines': {'visible': False}})
        chart.set_legend({'none': True})
        worksheet.insert_chart('N2', chart)
        
        chart = workbook.add_chart({'type': 'scatter'})
        chart.add_series({'name': 'S.Residual vs. Predicted',
                          'categories': sheet_name + '!$A$2:$A$' + str(data.shape[0]),
                          'values': sheet_name + '!$D$2:$D$' + str(data.shape[0])})
        chart.set_x_axis({'name': 'Predicted Value'})
        chart.set_y_axis({'name': 'Standardized Residual Value',
                          'major_gridlines': {'visible': False}})
        chart.set_legend({'none': True})
        worksheet.insert_chart('N17', chart)
        
        writer.save()
        
    def train(self,X,y):
        """Train the model object to get coefficient estimates.

        Arguments:
            
            X (Numpy Array) - An n-by-p array of floats depicting each observation
            (row) of each regressor (independent variable)(column).
            
            y (Numpy Array) - An n-by-1 array of floats depicting each 
            observation (row) of the response (dependent) variable.
                        
        Returns:
            
            None
            
        Raises Error:
            
            None
            
        """
        X = self.checkIntercept(X)
        self.beta = np.dot(la.inv(np.dot(X.T,X)),np.dot(X.T,y))
        self.regStatsTrn = self.RegStats(X,y,self.beta)
        
    def validate(self,X,y):
        """Validate the trained model on withheld data.

        Arguments:
            
            X (Numpy Array) - An n-by-p array of floats depicting each observation
            (row) of each regressor (independent variable)(column).
            
            y (Numpy Array) - An n-by-1 array of floats depicting each 
            observation (row) of the response (dependent) variable.
                        
        Returns:
            
            None
            
        Raises Error:
            
            None
            
        """
        X = self.checkIntercept(X)
        self.regStatsVal = self.RegStats(X,y,self.beta)
          
    class RegStats:
        """A subclass used to store useful regression statistics at various 
        stages in the execution of the OLS Regression."""
        def __init__(self,X = None,y = None,beta = None,proc = None):
            """Calculates appropriate regression statistics based on the
            results' origin process, stores them as model attributes."""
            if proc == 'cval':
                self.r2 = None
                self.mr = None
                self.r2_adj = None
                self.se = None
                self.aic = None
                self.bic = None
                self.rmse = None
            else:
                n = X.shape[0]
                p = X.shape[1]
        
                yhat = np.dot(X,beta)
                
                self.ssr = np.sum((y - yhat)**2)
                self.sst = np.sum((y - np.mean(y))**2)
                
                self.r2 = 1-(self.ssr/self.sst)
                if self.r2 < 0:
                    self.mr = None
                else:
                    self.mr = self.r2**(1/2)
                self.r2_adj = self.r2-(1-self.r2)*(p/(n-p-1))
                self.se = np.sqrt(self.ssr/(n-p-1))
                self.aic = 2*beta.shape[0] + (n*np.log(self.ssr/n))
                self.bic = beta.shape[0]*np.log(n) - 2*(np.log(self.ssr/n))
                self.rmse = (np.sum((yhat - y)**2))**(1/2)
           
                if proc == 'pred':
                    
                    self.sse = np.sum((yhat - np.mean(y))**2)
                    
                    self.df_reg = p
                    self.df_tot = n-1
                    self.df_res = n-p-1
                    
                    self.mse_reg = self.sse/self.df_reg
                    self.mse_res = self.ssr/self.df_res
                    self.fstat = (self.r2/self.df_reg)/((1-self.r2)/self.df_res) 
                    self.fsig = 1-f.cdf(self.fstat,self.df_reg,self.df_res)
                    self.se_coeff = np.sqrt(np.diag(self.mse_res*la.inv(np.dot(X.T,X))))
                    self.t_stat = abs(np.diag(np.divide(beta,self.se_coeff)))
                    self.p_vals = 2*(1-(t.cdf(self.t_stat, df = self.df_res)))
                    self.lower = np.diag(beta - (t.ppf(1-(0.05/2),df = self.df_res)*self.se_coeff))
                    self.upper = np.diag(beta + (t.ppf(1-(0.05/2),df = self.df_res)*self.se_coeff))
                    
if __name__ == '__main__':
    main()