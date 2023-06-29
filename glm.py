# -*- coding: utf-8 -*-

import statsmodels.api as sm
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt



def glm(X, y, x_test, y_test, x_train, y_train):
    
    # Add a constant term to the predictor variables
    # X = sm.add_constant(X)
    
    # Specify the GLM model and fit it to the data
    start = time.time()
    glm_model = sm.GLM(y_train.astype(float), sm.add_constant(x_train.astype(float)), family=sm.families.Gaussian())
    glm_results = glm_model.fit()
    end = time.time()
    print("training time",end - start)
    
    # Print the summary of the model
    print(glm_results.summary())
    
    # Get the predicted values
    y_pred = glm_results.predict(sm.add_constant(x_test))
    y_pred_2 = glm_results.predict(sm.add_constant(X))
    
    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate the mean absolute error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate the R-squared value
    # rsquared = glm_results.rsquared
    
    # Calculate the AIC and BIC values
    aic = glm_results.aic
    bic = glm_results.bic
    
    print("Evaluation metrics:")
    print("mse", mse)
    print("rmse", rmse)
    print("mae", mae)
    # print("r-squared", rsquared)
    print("aic", aic)
    print("bic", bic)
    
    # Plot the different scatterplots (separed in 3 groups)
    
    plt.scatter(X["Lagging_Current_Reactive_Power.kVarh"], y_pred_2, c = "green", marker = "v")
    plt.scatter(X['Leading_Current_Reactive_Power.kVarh'], y_pred_2, c = "red", marker = ",")
    plt.scatter(X['tCO2.ppm'], y_pred_2, c = "blue", marker = "o")
    
    plt.title("Energy consumption vs. Current Reactive Power and tCO2")
    plt.legend(["Lagging_Current_Reactive_Power.kVarh", 'Leading_Current_Reactive_Power.kVarh',\
                'tCO2.ppm'])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(-2.5, 2.7)
    plt.show()
    
    
    plt.scatter(X["Lagging_Current_Power_Factor"], y_pred_2, c = "black", marker = ".")
    plt.scatter(X['Leading_Current_Power_Factor'], y_pred_2, c = "orange", marker = "^")
    
    plt.title("Energy consumption vs. Lagging_Current_Power_Factor and Leading_Current_Power_Factor")
    plt.legend(["Lagging_Current_Power_Factor", 'Leading_Current_Power_Factor'])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(10, 100)
    plt.show()
    
    
    plt.scatter(X['NSM'], y_pred_2, c = "yellow", marker = "<")
    
    plt.title("Energy consumption vs. NSM")
    plt.legend(["NSM"])
    plt.ylabel("energy consumption")
    plt.xlabel("predictors")
    plt.xlim(-50,85000)
    plt.show()
    
    return