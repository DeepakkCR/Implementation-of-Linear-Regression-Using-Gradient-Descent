# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Define the hypothesis (model)
2.Define the cost function
3.Update parameters using gradient descent
4.Train and predict
   

## Program:
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Startup.csv")

# Select one feature (R&D Spend) and target (Profit)
X = data['R&D Spend'].values
y = data['Profit'].values

# Normalize (important for gradient descent)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions for plotting
y_pred = m * X + b

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()

~~~
```
/*
Program to implement the linear regression using gradient descent.
Developed by: LOKESH K
RegisterNumber:  212225240080
*/
```

## Output:
<img width="835" height="638" alt="WhatsApp Image 2026-04-29 at 3 25 28 PM" src="https://github.com/user-attachments/assets/a9c4f666-7554-4575-ac31-dba64d6f836b" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
