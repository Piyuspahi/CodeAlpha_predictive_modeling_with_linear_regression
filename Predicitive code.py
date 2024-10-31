# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

##load data framre 
data = pd.read_csv(r"C:\Users\Piyus Pahi\Documents\Code Alpha Data Science Project\Predictive Modeling\housing.csv")
data

##describe about the data
data.describe()

data.info()

data.columns

##check duplicates
data.duplicated().sum()

##check null values
data.isnull().sum()

# Data exploration: Plotting the correlation matrix
plt.figure(figsize=(9, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Boston Housing Data')
plt.show()

##split in to input and output
X = data[['RM']]  
y = data['MEDV']  

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Visualize the results: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.title('Actual vs Predicted Home Prices (using RM feature)')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Median Home Value (MEDV)')
plt.legend()
plt.show()








