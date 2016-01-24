# Getting all libraries that have dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# importing the data from csv files
train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')


train_data.head()

# Taking the price column from the pandas data frame
price = train_data.ix[:,2:3]


# Doing some basic calculations on the data array.
sum_of_prices = price.sum()
size = price.size
avg = sum_of_prices/size

print "The mean of price using method one is ", str(avg)
print train_data['price'].values.size
print train_data['sqft_living'].values.size


# Defining a function for a simple linear regression using the scikit-learn library
def simple_linear_regression(input_feature, output):

    # use the formula for the slope
    regr = linear_model.LinearRegression()
    model = regr.fit(input_feature, output)

    return model.intercept_, model.coef_


# Checking the output of the defined fiunction
input_feature = train_data['sqft_living']
output = train_data['price']

sqft_intercept, sqft_slope = simple_linear_regression(input_feature.reshape(-1,1), output)
print sqft_intercept, sqft_slope


# Estimating the price of the house for a given sq.ft area.
my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)


# Defining a function to get the resuduals.
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predictions = get_regression_predictions(input_feature, intercept, slope)

    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = predictions - output

    # square the residuals and add them up
    RSS = sum(residuals**2)
    return RSS
	
	

# Checking the residuals for a set of points.
rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)


# Defining a function for getting the inverse of the residuals.
def inverse_regression_predictions(output, intercept, slope):
    estimated_feature =  (output - intercept)/slope
    return estimated_feature

	
# Checking the inverse resudial function
my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)



# Checking the RSS based on the number of bedrooms on the TEST dataset.
input_feature = train_data['bedrooms']
output = train_data['price']

sqft_intercept, sqft_slope = simple_linear_regression(input_feature.reshape(-1,1), output)
rss_prices_on_bedrooms = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on bedrooms is : ' + str(rss_prices_on_bedrooms)


# Checking the RSS based onn the sq.ft area of the TEST dataset.
input_feature = train_data['sqft_living']
output = train_data['price']

sqft_intercept, sqft_slope = simple_linear_regression(input_feature.reshape(-1,1), output)
rss_prices_on_test_sqft = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on sqft is : ' + str(rss_prices_on_test_sqft)