# 📚 K-Nearest Neighbors (KNN) from Scratch

This folder contains a **from-scratch** implementation of the **K-Nearest Neighbors (KNN)** algorithm for **classification** and **regression** using only **NumPy** and basic Python.

The goal is to help students **understand how KNN works internally** — without relying on high-level libraries — and apply it to **real-world datasets**.

---

## 🔍 What is KNN?

K-Nearest Neighbors is a **supervised machine learning algorithm** used for:
- **Classification** → Predicting a class label
- **Regression** → Predicting a continuous value

It’s a **lazy learning algorithm**:  
- It doesn’t learn a parametric model during training.
- Instead, it stores the training data and makes predictions by comparing new data points to the stored ones.

---

## ⚙️ How KNN Works (Step-by-Step)

### **1. Choose `k`**
- `k` = the number of neighbors to consider.
- Small `k` → sensitive to noise.  
- Large `k` → smoother decision boundaries but may underfit.

---

### **2. Calculate Distance**
For a new point `x`, compute the distance to every training point.

**Common choices:**

- **Euclidean distance** (default in our code):  
  `distance(x, y) = sqrt( Σ (xᵢ - yᵢ)² )`

- **Manhattan distance**:  
  `distance(x, y) = Σ |xᵢ - yᵢ|`

---

### **3. Find the K Nearest Neighbors**
- Sort the training points by distance.
- Take the top `k` closest points.

---

### **4. Make Prediction**
- **Classification** → Majority vote among the k neighbors.  
- **Regression** → Average the target values of the k neighbors.

---

### **5. Evaluate**
- Classification → Accuracy, F1-score  
- Regression → RMSE, MAE

---

## 🧮 Complexity
- **Training time**: \( O(1) \) (just store the data)
- **Prediction time**: \( O(n \cdot d) \) where:
  - `n` = number of training samples
  - `d` = number of features

---

## 📦 Project Structure
