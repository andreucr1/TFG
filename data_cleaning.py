# -*- coding: utf-8 -*-

# Import packages to be used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mrm import mrm
from glm import glm
from rsm import rsm



# Read the data
steel_industry_df = pd.read_csv(r"C:\Users\pc\Desktop\TFG\Data\Steel_industry_data.csv")

# Remove useless column
steel_industry_df = steel_industry_df.drop(columns="WeekStatus")

# Standardize columns names
steel_industry_df.rename(columns={'date': 'Date', 'Usage_kWh': 'Usage.kWh',\
                                  'Lagging_Current_Reactive.Power_kVarh': 'Lagging_Current_Reactive_Power.kVarh',\
                                  'Leading_Current_Reactive_Power_kVarh': 'Leading_Current_Reactive_Power.kVarh', \
                                  'CO2(tCO2)': 'tCO2.ppm', 'Day_of_week': 'Day_Of_The_Week'}, inplace=True)

steel_industry_df_copy = steel_industry_df.copy()

# Check NaNs
number_of_nan = steel_industry_df.isnull().sum().sum()
print("The number of NaN characters in this dataset is:", number_of_nan)

# Shapiro-Wilk test


# Plot and compute skewness and kurtosis for the independent variables
num_cols = steel_industry_df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    print(col)
    print('Skewness :', round(steel_industry_df[col].skew(), 2))
    print('Kurtosis :', round(steel_industry_df[col].kurt(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    steel_industry_df[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=steel_industry_df[col])
    plt.show()
    
# Plot the correlation between variables
plt.figure(figsize=(12, 7))
sns.heatmap(steel_industry_df.drop(['Date', 'NSM', 'Day_Of_The_Week', 'Load_Type'],axis=1).corr(), annot = True, vmin = -1, vmax = 1)
plt.show()

# Apply Cube root Transformation to soften the skewness
num_cols.remove('NSM')
num_cols.remove('Lagging_Current_Power_Factor')
num_cols.remove('Leading_Current_Power_Factor')

for col in num_cols:
    steel_industry_df[col] = np.cbrt(steel_industry_df[col])        
    print(steel_industry_df[col].skew())
    
# Apply PCA to reduce collinearity
sc = StandardScaler()
X_scaled = sc.fit_transform(steel_industry_df[['Usage.kWh', 'Lagging_Current_Reactive_Power.kVarh', \
                                              'Leading_Current_Reactive_Power.kVarh', 'tCO2.ppm', \
                                              'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor']])

# Apply PCA
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# Get the transformed dataset
X_pca = pd.DataFrame(X_pca)

# Substitute the newly created columns to the original dataset
pca_cols = X_pca.columns.tolist()
for col, pca_col in zip(num_cols, pca_cols):
    steel_industry_df[col] = X_pca[pca_col]
    
plt.figure(figsize=(12, 7))
sns.heatmap(steel_industry_df.drop(['Date', 'NSM', 'Day_Of_The_Week', 'Load_Type'],axis=1).corr(), annot = True, vmin = -1, vmax = 1)
plt.show()


# Create the different models

train = steel_industry_df.sample(frac=0.8, random_state=1)
test = steel_industry_df.drop(train.index)

# Define the target variable and predictor variables

target_variable = 'Usage.kWh'
predictor_variables = ['Lagging_Current_Reactive_Power.kVarh',\
                       'Leading_Current_Reactive_Power.kVarh','Lagging_Current_Power_Factor',\
                       'Leading_Current_Power_Factor','tCO2.ppm', 'NSM']

y = steel_industry_df[target_variable]
X = steel_industry_df[predictor_variables]

# define dependent and independent variables
y_train = train[target_variable]
x_train = train[predictor_variables]

x_test = test[predictor_variables]
y_test = test[target_variable]


# MRM

mrm_rss_result = mrm(X, y)


# GLM

glm_result = glm(X, y, x_test, y_test, x_train, y_train)


# RSM

rsm_result = rsm(X, y)


# Extra information

# Means of the predictors
print('Lagging_Current_Reactive_Power.kVarh mean', steel_industry_df_copy['Lagging_Current_Reactive_Power.kVarh'].mean())
print('Leading_Current_Reactive_Power.kVarh mean', steel_industry_df_copy['Leading_Current_Reactive_Power.kVarh'].mean())
print('Lagging_Current_Power_Factor mean', steel_industry_df_copy['Lagging_Current_Power_Factor'].mean())
print('Leading_Current_Power_Factor mean', steel_industry_df_copy['Leading_Current_Power_Factor'].mean())
print('tCO2.ppm mean', steel_industry_df_copy['tCO2.ppm'].mean()) 
print('NSM mean', steel_industry_df_copy['NSM'].mean())

print("check:", steel_industry_df_copy.mean())