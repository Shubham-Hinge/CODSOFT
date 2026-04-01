# Titanic Survival Prediction  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

 #Load Dataset 
df = pd.read_csv("Titanic-Dataset.csv")

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head()) 
#Data Overview 
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())
 #Exploratory Data Analysis -Missing Values Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Class
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show() 
# Data Preprocessing  -Fill missing values  
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (if exists)
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# Drop unnecessary columns
df = df.drop(columns=['Name', 'Ticket', 'PassengerId'])

# Verify no missing values
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Final safety check
df = df.dropna()
 # Feature Engineering 
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
 # Encoding 
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked']) 
#Correlation Heatmap (after encoding) 
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show() 
# Feature & Target Split 
X = df.drop('Survived', axis=1)
y = df['Survived'] 
# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 
# Model Training 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train) 
#Predictions 
y_pred = model.predict(X_test)
 # Evaluation 
print("\nModel Evaluation")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
 # Feature Importance 
importance = pd.Series(model.coef_[0], index=X.columns)

plt.figure(figsize=(8,5))
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show() 
#Sample Prediction 
sample = X_test.iloc[0:1]

print("\nSample Input:\n", sample)
print("Predicted Survival:", model.predict(sample)[0])
 
