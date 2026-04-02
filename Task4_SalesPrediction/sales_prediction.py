# Sales Prediction using Machine Learning 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
df = pd.read_csv("advertising.csv")    

print("\nDataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)
#  Clean Column Names 
df.columns = df.columns.str.strip().str.lower()

print("\nColumns:", df.columns)
#  Data Cleaning 
# Drop unnamed index column if exists
if 'unnamed: 0' in df.columns:
    df = df.drop(columns=['unnamed: 0'])

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())
#  Visualization (EDA) 
# Pairplot (relationship)
sns.pairplot(df)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Individual relationships
plt.figure(figsize=(6,4))
sns.scatterplot(x='tv', y='sales', data=df)
plt.title("TV vs Sales")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='radio', y='sales', data=df)
plt.title("Radio vs Sales")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='newspaper', y='sales', data=df)
plt.title("Newspaper vs Sales")
plt.show()
# Feature & Target Split 
X = df[['tv', 'radio', 'newspaper']]
y = df['sales']
#  Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#  Model Training 
model = LinearRegression()
model.fit(X_train, y_train)
#  Prediction 
y_pred = model.predict(X_test)
# Evaluation 

print("\nModel Evaluation")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
#  Actual vs Predicted Plot 
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
# Feature Importance (Coefficients) 
coeff = pd.Series(model.coef_, index=X.columns)

plt.figure(figsize=(6,4))
coeff.sort_values().plot(kind='barh')
plt.title("Feature Importance (Impact on Sales)")
plt.show()
#  Sample Prediction 
sample = X_test.iloc[0:1]

print("\nSample Input:\n", sample)
print("Predicted Sales:", model.predict(sample)[0])
