
#Importing the required libraries

import pandas as pd

import numpy as np

#Here pandas is used to read the csv file and to convert it into a dataframe object

df = pd.DataFrame(pd.read_csv('....path/file.csv')

#path of the file has to be given as input

#now we have to choose the input X and target y 

X = df.iloc[:,:-1]    #this will select all the rows and with all the columns befrore the last column

y = df.iloc[:,-1]      # this will select all the rows with last column

#then we have to split the data into training and testing data such that 80% for training, 20% for testing data both X and y

#for this we need sklearn module

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)

#test_size=0.2 means test data is 20%.

#this random_state =45 tells a bit of intensity of randomization or say % of randomisation.

#now to perform KN-regression we have to import it and fit the training data to it

from sklearn.neighbors import KNeighborsRegressor

regr = KNeighborsRegressor()

regr.fit(X_train,y_train)

print(regr.predict([X_value]))  #input some X_value

#it's time to predict the X_test data

y_pred = regr.predict(X_test)

# to know the model performance we have to compare the y test data to predicted y data

#it is performed using r2_score from metrics class

from sklearn.metrics import r2_score

print(regr.score(X_test,y_test))

print(r2_score(y_test,y_pred))

#these two are same and will give same result

#furthur we can calculate the following statistical parameters

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
