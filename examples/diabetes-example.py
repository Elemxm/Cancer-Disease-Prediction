#Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#Data Dollection and Analysis
#PIMA Diabetes Dataset

#Loading the diabetes dataset to Pandas DataFrame
diabetes_dataset = pd.read_csv('examples/diabetes.csv')

#printing the first 5 rows of the dataset
print(diabetes_dataset.head())  

# Number of rows and Columns in this Dataset
print(diabetes_dataset.shape)

# Getting the Statictical measueres of the data
print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts()) #Find how many are Diabetics and how many are not

# Outcome = 1 -> Diabetic 
# Outcome = 0 -> Non Diabetic
print(diabetes_dataset.groupby('Outcome').mean())

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X) # All data exept from Outcome
print(Y) # Outcome

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
stadarized_data = scaler.transform(X)


'''
We did we include Data Standardization?
It ensures that features with larger numerical ranges don’t dominate those with smaller ranges.

It speeds up convergence in gradient-based models (like Logistic Regression or Neural Networks).

It makes distance-based models (like KNN or SVM) more accurate because it ensures all features contribute equally.
'''
print(stadarized_data)
X = stadarized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, stratify=Y, random_state=2)

# Training the Model
classifier = svm.SVC(kernel='linear')
# training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# Model Evaluation
#Accuracy Score on the trainig data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy Score on the testing Data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

# Making a predictive System
input_data = (4,110,92,0,0,37.6,0.191,30)
# 4,110,92,0,0,37.6,0.191,30,0 -> This person is Not a Diabetic one

input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the imput data
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not Diabetic')
else:
    print('The person is Diabetic')
