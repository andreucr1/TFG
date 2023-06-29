# -*- coding: utf-8 -*-

import statsmodels.api as sm
import numpy as np
from scipy.optimize import minimize
from scipy.stats import f
import matplotlib.pyplot as plt
import time

def rsm(X, y):
    # Create a design matrix for the RSM model
    X_np = X.to_numpy()

    # Define the response variable
    y_np = y.to_numpy()

    # Add quadratic and interaction terms to the design matrix
    X_squared = np.square(X_np)
    X_interact = np.zeros((X_np.shape[0], int(X_np.shape[1]*(X_np.shape[1]-1)/2)))
    k = 0
    for i in range(X_np.shape[1]-1):
        for j in range(i+1, X_np.shape[1]):
            X_interact[:,k] = X_np[:,i]*X_np[:,j]
            k += 1
    X_design = np.concatenate((X_np, X_squared, X_interact), axis=1)
    
    # Fit the RSM model using ordinary least squares regression
    start = time.time()
    model = sm.OLS(y_np, sm.add_constant(X_design)).fit()
    end = time.time()
    print("training time",end - start)
    
    # Print a summary of the RSM model
    print(model.summary())
    
    # Perform a first-order fit to estimate the initial beta values
    beta_init = [np.mean(y_np), 0, 0, 0, 0, 0]
    res = minimize(objective_function, beta_init, args=(X_np, y_np))
    beta_hat = res.x
    
    # Calculate the pure error and total sum of squares
    y_mean = np.mean(y_np)
    SSE = objective_function(beta_hat, X_np, y_np)
    SST = np.sum((y_np - y_mean)**2)
    
    # Calculate the regression sum of squares and the R-squared value
    SSR = SST - SSE
    R2 = SSR / SST
    print("values: ", SSE, SST, SSR)
    # Calculate the standard error of the estimate and the confidence interval
    n = len(y_np)
    p = len(beta_hat)
    MSE = SSE / (n - p)
    se = np.sqrt(MSE)
    alpha = 0.05
    
    # Perform an analysis of variance (ANOVA) to test for significant effects
    F_crit = f.ppf(1 - alpha, p, n - p)
    F_stat = SSR / (MSE * p)
    p_val = 1 - f.cdf(F_stat, p, n - p)
    print("F-statistic = %.3f, p-value = %.3f" % (F_stat, p_val))
    if F_stat > F_crit:
        print("At least one predictor is significant")
    else:
        print("No significant predictors")
    
    # Print the results
    print("Regression coefficients: ", beta_hat)
    print("R-squared value: %.3f" % R2)
    print("Standard error of estimate: %.3f" % se)
    
    # Plot the different scatterplots (separed in 3 groups)
    
    plt.scatter(X["Lagging_Current_Reactive_Power.kVarh"], y, c = "green", marker = "v")
    plt.scatter(X['Leading_Current_Reactive_Power.kVarh'], y, c = "red", marker = ",")
    plt.scatter(X['tCO2.ppm'], y, c = "blue", marker = "o")
    
    plt.title("Energy consumption vs. Current Reactive Power and tCO2")
    plt.legend(["Lagging_Current_Reactive_Power.kVarh", 'Leading_Current_Reactive_Power.kVarh',\
                'tCO2.ppm'])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(-2.5, 2.7)
    plt.show()
    
    
    plt.scatter(X["Lagging_Current_Power_Factor"], y, c = "black", marker = ".")
    plt.scatter(X['Leading_Current_Power_Factor'], y, c = "orange", marker = "^")
    
    plt.title("Energy consumption vs. Lagging_Current_Power_Factor and Leading_Current_Power_Factor")
    plt.legend(["Lagging_Current_Power_Factor", 'Leading_Current_Power_Factor'])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(10, 100)
    plt.show()
    
    
    plt.scatter(X['NSM'], y, c = "yellow", marker = "<")
    
    plt.title("Energy consumption vs. NSM")
    plt.legend(["NSM"])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(-50,85000)
    plt.show()

# Define the response surface model
def response_surface_model(x, beta):
    return beta[0] + beta[1]*x[0] + beta[2]*x[1] + beta[3]*x[0]**2 + beta[4]*x[1]**2 + beta[5]*x[0]*x[1]

# Define the objective function
def objective_function(beta, X, Y):
    y = [response_surface_model(x, beta) for x in X]
    mse = np.mean((y - Y)**2)
    return mse



