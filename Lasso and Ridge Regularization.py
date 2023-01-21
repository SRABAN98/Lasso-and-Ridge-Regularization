#Import The Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Import Linear Regression Machine Learning Libraries
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score


#Import The Dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\22nd\lasso, ridge, elastic net\TASK-22_LASSO,RIDGE\car-mpg (1).csv")


#Data Pre-processing
dataset = dataset.drop(["car_name"], axis = 1)
dataset["origin"] = dataset["origin"].replace({1: "america", 2: "europe", 3: "asia"})
dataset = pd.get_dummies(dataset,columns = ["origin"])
dataset = dataset.replace("?", np.nan)
dataset = dataset.apply(lambda x: x.fillna(x.median()), axis = 0)


#Divide The Dataset Into I.V and D.V
x = dataset.drop(["mpg"], axis = 1)          #Independent Variable
y = dataset[["mpg"]]                         #Dependent Variable


#Scale The Dataset(Feature Scalling)
x_s = preprocessing.scale(x)
x_s = pd.DataFrame(x_s, columns = x.columns)       #Converting Scaled Data Into Dataframe
y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns = y.columns)       #ideally train, test data should be in columns


#Train Test split
x_train, x_test, y_train, y_test = train_test_split(x_s, y_s, test_size = 0.30, random_state = 1)
x_train.shape
x_test.shape


#Fit Simple Linear Regression Model and Find Co-efficients
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)
for idx, col_name in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))    
intercept = regression_model.intercept_[0]
print("The intercept is {}".format(intercept))


#Fit Lasso Regularization Into The Dataset
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of co-efficient
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(x_train, y_train)
print("Lasso model coef: {}".format(lasso_model.coef_))
#As the data has 10 columns hence 10 co-efficients appear here


#Fit Ridge Regularization Into The Dataset
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of co-efficient
ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(x_train, y_train)
print("Ridge model coef: {}".format(ridge_model.coef_))
#As the data has 10 columns hence 10 co-efficients appear here


#Model Score - r^2 or co-efficient of determinant
#r^2 = 1-(SSR/SST) = Regression error/SST

#Simple Linear Model
print(regression_model.score(x_train, y_train))
print(regression_model.score(x_test, y_test))

print("*************************")

#Ridge Regularization
print(ridge_model.score(x_train, y_train))
print(ridge_model.score(x_test, y_test))

print("*************************")

#Lasso Regularization
print(lasso_model.score(x_train, y_train))
print(lasso_model.score(x_test, y_test))


#Polynomial Features
poly = PolynomialFeatures(degree = 2, interaction_only = True)
#Fit calculates mean and standard deviation while transform applies the transformation to a particular set of examples
#Here fit_transform helps to fit and transform the x_s
#Hence type(x_poly) is numpy.array while type(x_s) is pandas.DataFrame 
x_poly = poly.fit_transform(x_s)
#Similarly capture the co-efficients and intercepts of this polynomial feature model


#Model Parameter Tuning
data_train_test = pd.concat([x_train, y_train], axis =1)
data_train_test.head()


import statsmodels.formula.api as smf
ols1 = smf.ols(formula = "mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin_america+origin_europe+origin_asia", data = data_train_test).fit()
ols1.params


print(ols1.summary())
#Lets check Sum of Squared Errors (SSE) by predicting value of y for test cases and subtracting from the actual y for the test cases
mse  = np.mean((regression_model.predict(x_test)-y_test)**2)


# root of mean_sq_error is standard deviation i.e. avg variance between predicted and actual
import math
rmse = math.sqrt(mse)
print("Root Mean Squared Error: {}".format(rmse))


#Is OLS a good model ? Lets check the residuals for some of these predictor
fig = plt.figure(figsize=(10,8))
sns.residplot(x = x_test["hp"], y = y_test["mpg"], color = "green", lowess = True)


fig = plt.figure(figsize=(10,8))
sns.residplot(x = x_test["acc"], y = y_test["mpg"], color = "green", lowess = True )


#Predict mileage (mpg) for a set of attributes not in the training or test set
y_pred = regression_model.predict(x_test)


#Since this is regression, plot the predicted y value vs actual y values for the test data
#A good model's prediction will be close to actual leading to high R and R2 values
plt.rcParams["figure.dpi"] = 500
plt.scatter(y_test["mpg"], y_pred)


