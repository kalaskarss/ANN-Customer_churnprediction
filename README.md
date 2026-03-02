# 🏦 Bank Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

Customer churn prediction is a critical problem in the banking industry. Churn occurs when a customer stops using a bank’s services. Identifying customers who are likely to leave helps banks take proactive steps to retain them.

This project builds an Artificial Neural Network (ANN) model to predict whether a customer will churn based on historical customer data.

---

## 🎯 Objective

The objective of this project is to:

- Predict whether a bank customer will leave the bank.
- Help financial institutions reduce customer attrition.
- Improve retention strategies using machine learning.

---

## 📊 Dataset Description

The dataset contains bank customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target Variable)

### Target Variable:
- `1` → Customer Churned
- `0` → Customer Stayed

---

## 🧠 Model Architecture

The model is built using TensorFlow / Keras with the following architecture:

- Input Layer (11 features)
- Hidden Layer 1 (ReLU activation)
- Hidden Layer 2 (ReLU activation)
- Output Layer (Sigmoid activation)

### Model Configuration:
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Removed unnecessary columns
- Handled categorical variables (Label Encoding & One Hot Encoding)
- Feature Scaling using StandardScaler

### 2️⃣ Train-Test Split
- Split dataset into training and testing sets

### 3️⃣ Model Building
- Constructed ANN using Sequential API
- Added hidden layers
- Compiled the model

### 4️⃣ Model Training
- Trained model on training dataset
- Validated on test dataset

### 5️⃣ Model Evaluation
- Calculated Accuracy
- Generated Confusion Matrix
- Evaluated classification performance

---

## 📈 Model Performance

(Add your actual values here)

- Training Accuracy: 86 %
- Testing Accuracy: 84.5%
- Confusion Matrix
- Classification Report

## 🚀 How to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/kalaskarss/ANN-Customer_churnprediction.git
