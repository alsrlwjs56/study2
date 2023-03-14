import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocess the data

# Drop irrelevant features
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Impute missing values in the 'Age' feature using the median value
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
test_data['Age'] = imputer.transform(test_data[['Age']])

# Impute missing values in the 'Embarked' feature using the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
train_data['Embarked'] = imputer.fit_transform(train_data[['Embarked']])
test_data['Embarked'] = imputer.transform(test_data[['Embarked']])

# Impute missing values in the 'Fare' feature using the median value
imputer = SimpleImputer(strategy='median')
test_data['Fare'] = imputer.fit_transform(test_data[['Fare']])

# Encode categorical features using label encoding
encoder = LabelEncoder()
train_data['Sex'] = encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = encoder.transform(test_data['Sex'])

train_data['Embarked'] = encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = encoder.transform(test_data['Embarked'])

# Scale the features using StandardScaler
scaler = StandardScaler()
train_data[['Age', 'Fare']] = scaler.fit_transform(train_data[['Age', 'Fare']])
test_data[['Age', 'Fare']] = scaler.transform(test_data[['Age', 'Fare']])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop(['Survived'], axis=1), 
                                                  train_data['Survived'], test_size=0.2, random_state=42)

# Train a decision tree classifier on the training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the validation set and calculate the accuracy
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print('Accuracy:', acc)

# Make predictions on the test data and save the results
test_pred = clf.predict(test_data)
submission = pd.read_csv('gender_submission.csv')
submission['Survived'] = test_pred
submission.to_csv('titanic_predictions.csv', index=False)