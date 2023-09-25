import numpy as np
from sklearn.linear_model import LogisticRegression

# Loads the dataset
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Creates a logistic regression model
lr_model = LogisticRegression()
# Trains the model (fits it) with the dataset
lr_model.fit(X, y)

# Predicts the output using the trained model
y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)

# Predicts the accuracy 
print("Accuracy on training set:", lr_model.score(X, y))