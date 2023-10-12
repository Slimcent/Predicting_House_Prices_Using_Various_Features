import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate or load your dataset
data = {
    'SquareFootage': [1500, 2000, 1400, 1600, 2400, 1800, 2100, 1300],
    'Bedrooms': [3, 4, 3, 3, 4, 3, 4, 2],
    'Bathrooms': [2, 2.5, 2, 2, 3, 2, 3, 1.5],
    'DistanceToSchool': [0.5, 1.2, 0.8, 0.3, 1.5, 0.7, 1.0, 0.4],
    'AgeOfHouse': [5, 10, 8, 2, 15, 7, 12, 3],
    'Price': [250000, 320000, 270000, 260000, 400000, 290000, 350000, 230000]
}

df = pd.DataFrame(data)

# Split the dataset into training and testing sets
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the multiple linear regression model (same as before)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set (same as before)
y_pred = model.predict(X_test)

# Plot the regression line along with actual data points
plt.scatter(X_test['SquareFootage'], y_test, color='blue', label='Actual Prices')
plt.plot(X_test['SquareFootage'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Multiple Linear Regression')
plt.legend()
plt.show()
