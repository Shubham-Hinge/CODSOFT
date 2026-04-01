# Movie Rating Prediction  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
 # Load Dataset 
df = pd.read_csv("movie_data.csv", encoding='latin1')

print("\nDataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)
 
 #Clean Column Names 
df.columns = df.columns.str.strip()
 # Data Cleaning - Drop rows where Rating is missing
df = df.dropna(subset=['Rating'])

# Fill text columns
text_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for col in text_cols:
    df[col] = df[col].fillna("Unknown")


# ---- FIXED REGEX HERE ----
df['Year'] = df['Year'].str.extract(r'(\d+)')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')


# Clean Duration
df['Duration'] = df['Duration'].str.replace('min', '', regex=False)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')


# Clean Votes
df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')


# Fill numeric missing values
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())


# Final check
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Feature Engineering 
df['Combined'] = (
    df['Genre'] + " " +
    df['Director'] + " " +
    df['Actor 1'] + " " +
    df['Actor 2'] + " " +
    df['Actor 3']
)
 # Visualization 
plt.figure(figsize=(8,5))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title("Rating Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='Votes', y='Rating', data=df)
plt.title("Votes vs Rating")
plt.show()
 #TF-IDF Vectorization 
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df['Combined']).toarray()
 # Numerical Features 
X_num = df[['Year', 'Duration', 'Votes']].values


# Combine features
X = np.hstack((X_text, X_num))
y = df['Rating']
 #Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 # Model Training 
model = LinearRegression()
model.fit(X_train, y_train)
 # Prediction 
y_pred = model.predict(X_test)
 # Evaluation 

print("\nModel Evaluation")

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
 # Visualization 
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings")
plt.show()
 # Sample Prediction 
sample = X_test[0:1]
print("\nSample Prediction:", model.predict(sample))
