from flask import Flask, request, render_template, jsonify
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re


# Load the trained scm model and other necessary components
with open('models/svm', 'rb') as model_file:
    decision_tree_model = pickle.load(model_file)
#stopwords list
with open('models/fin_stop', 'rb') as f:
    stop = pickle.load(f)
#tf-idf instance that was fit on reviews corpus
with open('models/tv_fit', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Preprocessing function
def preprocess_review(review):
    stemmer=PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in stop]
    review = ' '.join(review)
    return review

#predict function
@app.route('/predict', methods=['POST'])
def predict():
    try:
        review = request.form['review']
        preprocessed_review = preprocess_review(review)
        review_vector = tfidf_vectorizer.transform([preprocessed_review]).toarray()
        prediction = decision_tree_model.predict(review_vector)
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})




if __name__ == '__main__':
    app.run(debug=True)
