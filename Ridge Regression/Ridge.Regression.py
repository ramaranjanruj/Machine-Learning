import pandas as pd
import numpy as np
from sklearn import linear_model
import math


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
            'sqft_living15':float, 'grade':int, 'yr_renovated':int,
            'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
            'sqft_lot15':float, 'sqft_living':float, 'floors':float,
            'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
            'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# Importing the sales data
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])


def polynomial_dataframe(feature, degree): # feature is pandas.Series type
"""
This function is to create polynomial features from a given column
"""
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = poly_dataframe['power_1'].apply(lambda x: math.pow(x, power))
    return poly_dataframe


poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
l2_small_penalty = 1.5e-5
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])
print model.coef_[0]  # Prints the coefficient of the column with power =1


"""
Reading the other 4 datasets
"""
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)


def get_ridge_coef(data, deg, l2_penalty):
    """
    This is a fucntion to generate the 1st coefficient of the ridge regression
    """
    poly15 = polynomial_dataframe(data['sqft_living'], deg)

    model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
    model.fit(poly15, data['price'])

    return model.coef_[0]


"""
Finf the ridge coefficients for the 4 datasets for l2_small_penalty=1e-9
"""
l2_small_penalty=1e-9

print get_ridge_coef(set_1, 15, l2_small_penalty)
print get_ridge_coef(set_2, 15, l2_small_penalty)
print get_ridge_coef(set_3, 15, l2_small_penalty)
print get_ridge_coef(set_4, 15, l2_small_penalty)




"""
Finf the ridge coefficients for the 4 datasets for l2_large_penalty=1.23e2
"""

l2_large_penalty=1.23e2

print get_ridge_coef(set_1, 15, l2_large_penalty)
print get_ridge_coef(set_2, 15, l2_large_penalty)
print get_ridge_coef(set_3, 15, l2_large_penalty)
print get_ridge_coef(set_4, 15, l2_large_penalty)



train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)


def k_fold_cross_validation(k, l2_penalty, data, output):
    """
    Function to find the best value of lambda in a given k_fold_cross_validation
    """
    n = len(data)
    total_mse = []
    best_mse = None
    best_lambda = 0

    poly_data = polynomial_dataframe(data['sqft_living'], 15)

    for l2_value in l2_penalty:
        for i in xrange(k):

            # Generates the index of the dataframe
            start = (n*i)/k
            end = (n*(i+1))/k-1
            round_mse = 0

            # Splits the dataframe
            X_test, Y_test = poly_data[start:end+1], output[start:end+1]
            X_train = pd.concat([poly_data[0:start], poly_data[end+1:n]], axis=0, ignore_index=True)
            Y_train = pd.concat([pd.Series(output[0:start]), pd.Series(output[end+1:n])], axis=0, ignore_index=True)

            # Peform a ridge regression of the training and testing sets
            ridge = linear_model.Ridge(alpha=l2_value, normalize=True)
            ridge.fit(X_train, Y_train)
            out = ridge.predict(X_test)
            round_mse += ((out - Y_test)**2).sum()

        # Calculate the mse for each value of lambda
        round_mse = round_mse/k

        # Get the best value of mse and lambda
        if best_mse is None or best_mse > round_mse:
            best_mse = round_mse
            best_lambda = l2_value

        total_mse.append(round_mse)

    return np.mean(total_mse), best_lambda

"""
Get the best value of lambda and mse on the kfolded training sets
"""

l2_penalty = np.logspace(3, 9, num=13)
kfold_mse, kfold_lambda = k_fold_cross_validation(10, l2_penalty, train_valid_shuffled, train_valid_shuffled['price'])

print kfold_mse         # 3.13176238516e+13
print kfold_lambda      # 1000.0



"""
Get the RSS on the testing datasets
"""

ridge_all = linear_model.Ridge(alpha=1000, normalize=True)
ridge_all.fit(polynomial_dataframe(train_valid_shuffled['sqft_living'],15), train_valid_shuffled['price'])
predicted = ridge_all.predict(polynomial_dataframe(test['sqft_living'],15))
print ((predicted - test['price'])**2).sum()        # 2.83856861224e+14
