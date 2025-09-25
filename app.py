from flask import Flask, request, jsonify, render_template
import pickle
import string
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('stopwords')

dataset = pd.read_csv(r"E:\Zain collage file\SEM V\spam\spam detect\dataset\emails-new.csv")
print(dataset.shape)

# Show dataset head (first 5 records)
print(dataset.head())

# Show dataset info
dataset.info()

# Show dataset statistics
dataset.describe()

# Check for missing data for each column 
print(dataset.isnull().sum())

# Check for duplicates and remove them 
dataset.drop_duplicates(inplace=True)

def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# Load the vectorizer object, NOT the transformed matrix
vectorizer_path = r"E:\Zain collage file\SEM V\spam\spam detect\models\vectorizer.pkl"

if os.path.exists(vectorizer_path):
    # Load the CountVectorizer object
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    print("vectorizer loaded")
else:    
    # Create and fit a new CountVectorizer if it doesn't exist
    vectorizer = CountVectorizer(analyzer=process)
    vectorizer.fit(dataset['text'])
    
    # Save the vectorizer object
    with open(vectorizer_path, "wb") as file:
        dump(vectorizer, file)
    print("Vectorizer saved in model")

# Use the vectorizer to transform the dataset text
# This will produce a csr_matrix
# print("producing csr matrix")
# message = vectorizer.transform(dataset['text'])

# X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.20, random_state=0)
# print("Completed splitting training and test dataset")

# Load the trained model and vectorizer
if(os.path.exists(r"E:\Zain collage file\SEM V\spam\spam detect\models\model.pkl")):
    model = pickle.load(open(r"E:\Zain collage file\SEM V\spam\spam detect\models\model.pkl", 'rb'))
    print("completed reading model")
else : 
    # Model creation
    model = MultinomialNB()
    # Model training
    # model.fit(X_train, y_train)
    # print("model trained")
    # Model saving
    dump(model, open(r"E:\Zain collage file\SEM V\spam\spam detect\models\model.pkl", 'wb'))
    print("model saved")

# # Model predictions on test set
# y_pred = model.predict(X_test)

# # Model Evaluation | Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("accuracy = ", accuracy*100)

# # Model predictions on test set
# y_pred = model.predict(X_test)

# # Model Evaluation | Classification report
# print(classification_report(y_test, y_pred))

#vectorizer = pickle.load(open(vectorizer_path, 'rb'))


app = Flask(__name__)

@app.route('/')
def index():
    # Serve the HTML frontend
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_content = data['email']
    
    # Preprocess the email and make predictions
    email_vector = vectorizer.transform([email_content])
    prediction = model.predict(email_vector)[0]
    
    # Determine if it's spam or not
    result = "Spam" if prediction == 1 else "Not Spam"
    
    # Send the result back to the frontend
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
