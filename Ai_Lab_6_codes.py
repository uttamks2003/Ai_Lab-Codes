# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define the dataset snippet
data =pd.read_csv("/content/drive/MyDrive/AIQUIZ/car.data")

# Convert the string data to a pandas DataFrame
# data = pd.DataFrame([x.split(',') for x in data.strip().split('\n')])

# Assign column names
data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Convert categorical variables to numeric encoding using LabelEncoder
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Separate features and target variable
X = data.drop('class', axis=1)
y = data['class']

# Function to perform the lab assignment challenges
def lab_assignment(X, y, test_size, criterion, n_repeats):
    accuracy_scores = []
    f_scores = []

    for _ in range(n_repeats):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        # Initialize the DecisionTreeClassifier with the given criterion
        clf = DecisionTreeClassifier(criterion=criterion)

        # Train the classifier
        clf.fit(X_train, y_train)

        # Predict the class labels for the test set
        y_pred = clf.predict(X_test)

        # Calculate the confusion matrix and F-score
        cm = confusion_matrix(y_test, y_pred)
        fscore = f1_score(y_test, y_pred, average='weighted')

        # Calculate the accuracy
        accuracy = (cm.diagonal().sum()) / cm.sum()
        accuracy_scores.append(accuracy)
        f_scores.append(fscore)

    # Calculate the average accuracy and average F-score
    average_accuracy = np.mean(accuracy_scores)
    average_fscore = np.mean(f_scores)
    return average_accuracy, average_fscore

# Perform the lab assignment challenges
test_sizes = [0.4, 0.3, 0.2]  # Corresponding to 60%, 70%, and 80% training data
criteria = ['entropy', 'gini']
n_repeats = 20

# Store the results
results = {}

for test_size in test_sizes:
    for criterion in criteria:
        key = f'{int((1-test_size)*100)}% training data with {criterion}'
        results[key] = lab_assignment(X, y, test_size, criterion, n_repeats)

# Print the results
for key, value in results.items():
    print(f"{key}: Average Accuracy: {value[0]}, Average F-score: {value[1]}")

# Explain overfitting with an example
print("\nOverfitting Example:")
print("If a decision tree is trained on a dataset with noise and it learns the noise as if it were a true pattern,")
print("it will make incorrect predictions when presented with new data. This is because it has learned the specific")
print("details of the training data, rather than the underlying patterns that could generalize to new data.")