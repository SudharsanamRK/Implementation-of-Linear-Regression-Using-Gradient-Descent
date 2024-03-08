# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe. 
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
```
## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Sudharsanam R K
RegisterNumber:  212222040163
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
def linear_regression (X1, y, learning_rate=0.01, num_iters=1000): 
    # Add a column of ones to X for the intercept term
  X = np.c_[np.ones(len(X1)), X1]

  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)

  # Perform gradient descent 
  for _ in range(num_iters):
    # Calculate predictions
    predictions = (X).dot (theta).reshape(-1, 1)

    # Calculate errors
    errors = (predictions - y).reshape(-1,1)

    # Update theta using gradient descent
    theta -= learning_rate * (1 / len(X1)) * X.T.dot (errors)

  return theta
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1, 1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1, 1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction = prediction.reshape(-1, 1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```

## Output:
![startup_csv file](https://github.com/SudharsanamRK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115523484/e474c78f-840e-477e-91f6-fb73f1c7c953)
![Predicted Value](https://github.com/SudharsanamRK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115523484/53886a4a-d489-4fe0-8225-27a3a10f5a1a)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
