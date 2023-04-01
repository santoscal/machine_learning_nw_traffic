#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import time
#import os module and allow user select a file.

# Load traffic capture CSV file
df = pd.read_csv('/home/ted/Desktop/int/4.2/project/datasets/dataset.train.csv')

df.head(5)


# In[3]:



# Extract relevant features
features = ["source.port", "protocol", "packet.size"]
X = df[features]

# Define the target variable using "packet.size" column
y = df["packet.size"]

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the feature data and transform the feature data
X_norm = scaler.fit_transform(X)

# Split the data into training and testing sets with a 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Create a Random Forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)

# print("Training model...")
# Define the loading animation
def loading_animation():
    chars = "/—\|" # The characters used for the animation
    index = 0
    while True:
        # Print the next character in the animation sequence
        sys.stdout.write('\r' + "Model Training... " + chars[index % len(chars)])
        sys.stdout.flush()
        index += 1
        time.sleep(0.1)

# Start the loading animation in a separate thread
import threading
loading_thread = threading.Thread(target=loading_animation)
loading_thread.daemon = True
loading_thread.start()
# Train the Random Forest classifier on the training data
rf.fit(X_train, y_train)


# Use the trained model to make predictions on the testing data
y_pred = rf.predict(X_test)


# Calculate the accuracy, precision, recall, and F1 score of the model
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


# # In[5]:


# # !ls

# # Load the new dataset into a Pandas DataFrame
# new_df = pd.read_csv('/home/ted/Desktop/int/4.2/project/datasets/dataset.train.csv')

# # Extract the relevant features from the new dataset
# X_new = new_df[features]

# # Preprocess the new data using the same normalization used during the training phase
# X_new_norm = scaler.transform(X_new)

# # Use the trained model to predict whether the new data contains a DDoS attack or not.
# predictions = rf.predict(X_new_norm)

# for prediction in predictions:
#     if prediction == 1:
#         print("DDoS attack detected!")
#     else:
#         print("No DDoS attack detected.")

