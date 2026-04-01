# Iris Flower Classification 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load Dataset 
df = pd.read_csv("Iris.csv")

print("\nDataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)
#Fix Column Names - Convert all columns to lowercase
df.columns = df.columns.str.strip().str.lower()

print("\nColumns:", df.columns)
#  Data Cleaning 
# Drop 'id' if exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())
#Visualization # Pairplot using correct column name
sns.pairplot(df, hue='species')
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df)
plt.title("Feature Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
#  Encoding Target 
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

class_names = le.classes_
#  Feature & Target Split 
X = df.drop('species', axis=1)
y = df['species']
#  Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#  Model Training 
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Prediction 
y_pred = model.predict(X_test)
# Evaluation 
print("\nModel Evaluation")

print("Accuracy:", accuracy_score(y_test, y_pred))

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
print(classification_report(y_test, y_pred, target_names=class_names))
#  Feature Importance 
importance = pd.Series(model.coef_[0], index=X.columns)

plt.figure(figsize=(8,5))
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()
# Sample Prediction 
sample = X_test.iloc[0:1]

predicted_class = model.predict(sample)[0]
print("\nSample Prediction:", class_names[predicted_class])
 
