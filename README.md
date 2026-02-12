# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare the dataset with input features and target values, and perform feature scaling.

2.Initialize the SGD regression model (manually with weights or using SGDRegressor). 

3.Train the model using stochastic gradient descent.

4.Use the trained model to predict and display the output values 

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: A.jaychandran
RegisterNumber:212225230112

#Manual Implementation using Numpy
import numpy as np
# Step 1: Sample dataset
# Features: [Hours Studied, Attendance, Previous Marks]
X = np.array([
[2, 80, 50],
[3, 60, 40],
[5, 90, 70],
[7, 85, 80],
[9, 95, 90]
], dtype=float)
# Target: Marks Scored
y = np.array([50, 45, 70, 80, 95], dtype=float)
# Step 2: Feature normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X] # shape becomes (n_samples, n_features + 1)
# Step 3: Initialize weights
n_features = X.shape[1]
weights = np.zeros(n_features)
# Hyperparameters
learning_rate = 0.01
epochs = 1000
# Step 4: Stochastic Gradient Descent
for epoch in range(epochs):
for i in range(X.shape[0]):
xi = X[i]
yi = y[i]
y_pred = np.dot(xi, weights)
error = y_pred - yi
# Update weights
weights -= learning_rate * error * xi
print("Trained Weights (including intercept):", weights)
# Step 5: Make predictions
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)
#Using scikit-learn SGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
# Features and target
X = np.array([
[2, 80, 50],
[3, 60, 40],
[5, 90, 70],
[7, 85, 80],
[9, 95, 90]
])
y = np.array([50, 45, 70, 80, 95])
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Create SGD Regressor
sgd_reg = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01,
random_state=42)
sgd_reg.fit(X_scaled, y)
# Coefficients and intercept
print("Weights (coefficients):", sgd_reg.coef_)
print("Intercept:", sgd_reg.intercept_)
# Predictions
y_pred = sgd_reg.predict(X_scaled)
print("Predicted values:", y_pred)
```

## Output:
![WhatsApp Image 2026-02-12 at 7 57 04 PM](https://github.com/user-attachments/assets/eb547042-4e06-4919-8614-0331af459be6)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
