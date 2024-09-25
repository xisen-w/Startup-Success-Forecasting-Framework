import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Assuming `df` is your DataFrame after loading the data with each row representing the structured data you provided



# Features and labels
X = df.drop('Outcome', axis=1)  # Drop the outcome column to separate features
y = df['Outcome']  # Assuming you have an 'Outcome' column with 'Successful' or 'Unsuccessful'

# Encoding categorical features
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

