# linear regression test

import pandas as pd
import matplotlib.pyplot as plt
from linear_regresion.LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# file path
PATH = '../dataset/USA_Housing.csv'

# load data
df = pd.read_csv(PATH)
df = df.drop('Address', axis=1)

X = df[['Avg. Area Income']].values
Y = df['Price'].values


# scale features for better gradient descent
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

model = LinearRegression(learning_rate=0.01, number_of_iterations=1000)

print("Training custom Linear Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

y_pred_custom = model.predict(X_test)

# Convert back to original scale for visualization
X_test_original = scaler.inverse_transform(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_original, y_test, color='blue', label='Actual Prices', alpha=0.5)


# sort data points for smooth regression line
sorted_indices = np.argsort(X_test_original.flatten())
plt.plot(X_test_original[sorted_indices], y_pred_custom[sorted_indices], color='red', linewidth=2, label='Predicted Prices (Regression Line)')

plt.title('Linear Regression: Avg. Area Income vs. Price')
plt.xlabel('Avg. Area Income')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
