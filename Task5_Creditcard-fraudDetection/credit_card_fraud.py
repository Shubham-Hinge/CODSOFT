# Credit Card Fraud Detection 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
 
df = pd.read_csv("creditcard.csv")

print("\nDataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)
 
print("\nClass Distribution:")
print(df['Class'].value_counts())
#  Visualization  
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.show()
#  Data Preprocessing 
# Scale Amount & Time
scaler = StandardScaler()

df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])
#  Feature & Target Split 
X = df.drop('Class', axis=1)
y = df['Class']
#  Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#  Handle Imbalance   
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_resampled).value_counts())
#  Model Training 
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)
#  Prediction 
y_pred = model.predict(X_test)
#  Evaluation 
print("\nModel Evaluation")

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#  Sample Prediction 
sample = X_test.iloc[0:1]

print("\nSample Prediction:", model.predict(sample)[0])
 

 
