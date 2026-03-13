# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.
2.Scale the features using a standard scaler to normalize the data.
3.Initialize model parameters (theta) and add an intercept term to the feature set.
4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.
5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S LOGESHWAR
RegisterNumber: 25014493 
*/

import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initialize parameters
m = 0  # slope
b = 0  # intercept

learning_rate = 0.01
epochs = 1000
n = len(x)

# Gradient Descent
for i in range(epochs):
    
    # Predicted values
    y_pred = m * x + b
    
    # Calculate gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Plot results
y_pred = m * x + b

plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, y_pred, color='red', label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
```

## Output:
<img width="658" height="476" alt="image" src="https://github.com/user-attachments/assets/6b14d805-bbca-4b4c-a316-ce7c4f023f7b" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
