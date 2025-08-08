import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from knn import KNN

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = KNN(k=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))



pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

colors = np.array(["red", "blue"])
correct_mask = y_pred == y_test

plt.figure(figsize=(8, 6))

plt.scatter(
    X_train_2d[:, 0], X_train_2d[:, 1],
    c=colors[y_train],
    alpha=0.3, label="Train data"
)


plt.scatter(
    X_test_2d[correct_mask, 0], X_test_2d[correct_mask, 1],
    c=colors[y_pred[correct_mask]],
    marker='o', edgecolor='k', s=80, label="Correct predictions"
)

plt.scatter(
    X_test_2d[~correct_mask, 0], X_test_2d[~correct_mask, 1],
    c=colors[y_pred[~correct_mask]],
    marker='x', s=80, label="Incorrect predictions"
)

plt.title("KNN Classification")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
