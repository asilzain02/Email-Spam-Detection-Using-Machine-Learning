📧 Spam Email Detection (Flask + ML)

A Flask web application for detecting spam emails using Natural Language Processing (NLP) and Machine Learning (Naive Bayes).
This project allows users to input an email, and the trained model predicts whether it’s Spam or Not Spam.

🌟 Features

✅ Preprocessing with NLTK (stopwords removal, punctuation removal).

✅ Vectorization using CountVectorizer.

✅ Machine Learning model: Multinomial Naive Bayes.

✅ Flask backend with REST API (/predict).

✅ HTML frontend (index.html) for user interaction.

✅ Model and Vectorizer saved as .pkl for reusability.

🛠 Tech Stack

Python 3.9+

Flask (web framework)

Scikit-learn (ML model & vectorization)

NLTK (stopwords processing)

Pandas (data handling)

Pickle (model persistence)

📂 Project Structure
<img width="830" height="330" alt="image" src="https://github.com/user-attachments/assets/a76b6d63-b83c-4f00-bc21-e77741962588" />


⚡ Installation
1️⃣ Clone Repository
git clone (https://github.com/asilzain02/Spam-Email-Detection-Flask-Machine-Learning-.git)

cd Spam-Email-Detection-Flask-Machine-Learning-

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


requirements.txt

Flask
scikit-learn
pandas
nltk

4️⃣ Download NLTK Stopwords
python -m nltk.downloader stopwords

🚀 Run the App
python app.py


Visit http://127.0.0.1:5000/
 in your browser.

🔮 API Usage
Endpoint: /predict

Method: POST
Request Body (JSON):

{
  "email": "Congratulations! You've won a free iPhone. Click here."
}


Response:

{
  "result": "Spam"
}

📊 Model Training

Dataset: emails-new.csv

Preprocessing: punctuation removal + stopwords removal.

Vectorization: CountVectorizer with custom analyzer.

Classifier: MultinomialNB

📌 Future Improvements

🔹 Add TF-IDF Vectorizer support.

🔹 Train with larger datasets.

🔹 Improve frontend with Bootstrap or React.

🔹 Deploy to Heroku / Render / AWS.

👨‍💻 Author

Developed by Asil Zain ✨
Contributions are welcome!
