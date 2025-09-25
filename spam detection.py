import pickle
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
import string
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('stopwords')
import os
from flask import Flask, request, jsonify


# Load the dataset
dataset = pd.read_csv(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\dataset\emails-new.csv')
print(dataset.shape)

# Show dataset head (first 5 records)
print(dataset.head())

# Show dataset info
dataset.info()

# Show dataset statistics
dataset.describe()

# # Visualize spam frequenices
# plt.figure(dpi=100)
# sns.countplot(dataset['spam'])
# plt.title("Spam Freqencies")
# plt.show()



# Check for missing data for each column 
print(dataset.isnull().sum())

# Check for duplicates and remove them 
dataset.drop_duplicates(inplace=True)

# Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

if(os.path.exists(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\vectorizer.pkl')):
    message = pickle.load(open(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\vectorizer.pkl', "rb"))
else:    
    # Fit the CountVectorizer to data
    message = CountVectorizer(analyzer=process).fit_transform(dataset['text'])

    # Save the vectorizer
    dump(message, open(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\vectorizer.pkl', "wb"))
    print("Saving vectorizer in models")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.20, random_state=0)
print("Completed splitting training and test dataset")

if(os.path.exists(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\model.pkl')):
    model = pickle.load(open(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\model.pkl', 'rb'))
    print("completed reading model")
else : 
    # Model creation
    model = MultinomialNB()
    # Model training
    model.fit(X_train, y_train)
    print("model trained")
    # Model saving
    dump(model, open(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\model.pkl', 'wb'))
    print("model saved")

# Model predictions on test set
y_pred = model.predict(X_test)

# Model Evaluation | Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("accuracy = ", accuracy*100)

# Model predictions on test set
y_pred = model.predict(X_test)

# Model Evaluation | Classification report
print(classification_report(y_test, y_pred))
#print(classification_report)

# Model Evaluation | Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(dpi=100)
sns.heatmap(cm, annot=True)
plt.title("Confusion matrix")
plt.show()

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if not data or 'email' not in data:
#         return jsonify({'error': 'Invalid input'}), 400

#     # Load vectorizer and model
#     vectorizer = pickle.load(open(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\vectorizer.pkl', 'rb'))
#     model = pickle.load(open(r'E:\Zain collage file\SEM V\Email-Spam-Detection-Using-NLP-main\model\model.pkl', 'rb'))

#     # Preprocess and predict
#     email_text = data['email']
#     email_vector = vectorizer.transform([email_text])
#     prediction = model.predict(email_vector)
#     prediction_label = 'Spam' if prediction[0] == 1 else 'Not Spam'

#     return jsonify({'prediction': prediction_label})

# if __name__ == "__main__":
#     app.run(debug=True)


