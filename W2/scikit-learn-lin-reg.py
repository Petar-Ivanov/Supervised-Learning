import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
'''from lab_utils_multi import  load_house_data
from lab_utils_common import dlc'''

np.set_printoptions(precision=2)

# Loading the data
'''X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']'''
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Feature scalling the dataset using scikit-learn
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

# Creates and fits a gradient descent linear regression model
sgdr = SGDRegressor(max_iter=1000) # creates the model and sets maximum iterations for training it
sgdr.fit(X_norm, y_train) # trains (fits) the model with the training data (inputs and outputs)

# Gets the weights and the bias of the model
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")

# Makes a prediction using the model
y_pred_sgd = sgdr.predict(X_norm) # OPTION 1: using the inbuilt predict function
y_pred = np.dot(X_norm, w_norm) + b_norm   # OPTION 2: writing the model by hand and using the weights and the bias that were calcutated during training
for i in range(len(y_train)):
    print(f"predicted: {y_pred_sgd[i]:0.2f}, actual: {y_train[i]}")

