import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 3]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)



# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("coefficients: ", regr.coef_)
print("intercept: ", regr.intercept_)
plt.scatter(diabetes_X_test,diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_pred)
plt.show()



