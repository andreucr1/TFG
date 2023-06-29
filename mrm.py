# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time


def mrm(X, y):
    
    X = sm.add_constant(X)
    
    start = time.time()
    # create the regression model
    model = sm.OLS(y, X).fit()
    end = time.time()
    print("training time",end - start)
    
    # calculate the predicted values
    y_pred = model.predict(X)
    
    # calculate the RSS
    RSS = np.sum((y - y_pred)**2)
    
    # calculate the ESS
    ESS = np.sum((y_pred - np.mean(y))**2)
    
    MSE = np.mean(model.resid**2)
    
    y_mean = np.mean(y)
    SSB = np.sum((y_pred - y_mean)**2)
    
    SSE = np.sum((y - y_pred)**2)

    # Initialize a dictionary to store the Mallows' Cp values for each model
    Cp_values = {}
    
    # Loop over the number of predictors
    for p in range(1, len(X.columns) + 1):
        
        # Fit the model with p predictors
        model_p = sm.OLS(y, sm.add_constant(X.iloc[:, :p])).fit()
        
        # Calculate the sum of squared errors for the model with p predictors
        SSEp = np.sum(model_p.resid**2)
        
        # Calculate Mallows' Cp for the model with p predictors
        Cp = (SSEp / MSE) - (len(y) - 2*p)
        
        # Store the Cp value in the dictionary
        Cp_values[p] = Cp
    
    # Print the dictionary of Cp values
    print(Cp_values)
    
    # Print the RSS and ESS
    print("RSS: ", RSS)
    print("ESS: ", ESS)
    print("SSB: ", SSB)
    print("SSE: ", SSE)
    print(model.summary())
    
    # Plot the different scatterplots (separed in 3 groups)
    
    plt.scatter(X["Lagging_Current_Reactive_Power.kVarh"], y_pred, c = "green", marker = "v")
    plt.scatter(X['Leading_Current_Reactive_Power.kVarh'], y_pred, c = "red", marker = ",")
    plt.scatter(X['tCO2.ppm'], y_pred, c = "blue", marker = "o")
    
    plt.title("Energy consumption vs. Current Reactive Power and tCO2")
    plt.legend(["Lagging_Current_Reactive_Power.kVarh", 'Leading_Current_Reactive_Power.kVarh',\
                'tCO2.ppm'])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(-2.5, 2.7)
    plt.show()
    
    
    plt.scatter(X["Lagging_Current_Power_Factor"], y_pred, c = "black", marker = ".")
    plt.scatter(X['Leading_Current_Power_Factor'], y_pred, c = "orange", marker = "^")
    
    plt.title("Energy consumption vs. Lagging_Current_Power_Factor and Leading_Current_Power_Factor")
    plt.legend(["Lagging_Current_Power_Factor", 'Leading_Current_Power_Factor'])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(10, 100)
    plt.show()
    
    
    plt.scatter(X['NSM'], y_pred, c = "yellow", marker = "<")
    
    plt.title("Energy consumption vs. NSM")
    plt.legend(["NSM"])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(-50,85000)
    plt.show()
    
    
    return




    