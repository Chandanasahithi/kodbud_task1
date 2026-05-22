# Step 1: Upload ZIP File
from google.colab import files

uploaded = files.upload()

# Step 2: Extract ZIP File
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

print("ZIP Extracted Successfully!")


# Step 3: Check Files Inside Dataset Folder
import os

print(os.listdir('dataset'))

# --------------------------------

# Step 4: Import Libraries
import pandas as pd
import numpy as np

# --------------------------------

# Step 5: Load Dataset
# Replace spam.csv if your file name is different

df = pd.read_csv('dataset/spam.csv', encoding='latin-1')

# --------------------------------

# Step 6: Keep Only Required Columns

df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# --------------------------------

# Step 7: Convert Labels to Numbers

df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# --------------------------------

# Step 8: Features and Labels

X = df['message']
y = df['label_num']

# --------------------------------

# Step 9: Split Dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------

# Step 10: Convert Text into Numbers

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# --------------------------------

# Step 11: Train Model

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

# --------------------------------

# Step 12: Predictions

y_pred = model.predict(X_test_vectorized)

# --------------------------------

# Step 13: Accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# --------------------------------

# Step 14: Test Custom Messages

while True:

    msg = input("\nEnter a message (or type 'exit' to stop): ")

    if msg.lower() == 'exit':
        print("Program Ended")
        break

    msg_vectorized = vectorizer.transform([msg])

    prediction = model.predict(msg_vectorized)

    if prediction[0] == 1:
        print("Spam Message")
    else:
        print("Not Spam")

# --------------------------------

# Step 15: Accuracy Graph (Optional)

import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.bar(['Accuracy'], [accuracy * 100])

plt.ylabel('Percentage')

plt.title('Spam Classifier Accuracy')

plt.ylim(0, 100)

plt.show()