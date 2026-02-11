# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and create the dataset as a pandas DataFrame.
2. Separate the independent variables (Age, Salary, YearsAtCompany, JobSatisfaction) and dependent variable (Churn).
3. Create and train a Decision Tree Classifier using the entropy criterion.
4. Predict the test results, evaluate accuracy, and visualize the decision tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Tanushree G
RegisterNumber: 212225040462
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = {
    "Age": [22, 35, 45, 28, 50, 41, 30, 26, 48, 33],
    "Salary": [20000, 50000, 70000, 30000, 90000, 65000, 40000, 25000, 85000, 48000],
    "YearsAtCompany": [1, 7, 15, 2, 20, 10, 4, 1, 18, 6],
    "JobSatisfaction": [2, 4, 3, 2, 5, 4, 3, 1, 5, 3],
    "Churn": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)

X = df[["Age", "Salary", "YearsAtCompany", "JobSatisfaction"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Age", "Salary", "YearsAtCompany", "JobSatisfaction"],
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree Classifier for Employee Churn Prediction")
plt.show()
```

## Output:
<img width="1083" height="791" alt="image" src="https://github.com/user-attachments/assets/6c2014bd-dd5d-4ba5-b8d6-0714f72aa1a8" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
