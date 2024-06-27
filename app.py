from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


with open('countvectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

with open('vector_matrix.pkl', 'rb') as file:
    vector_matrix = pickle.load(file)

with open('volunteering_data.pkl', 'rb') as file:
    volunteering_data = pickle.load(file)


def recommend_opportunities(user_input, data, vectorizer, vector_matrix, top_n=7):
    user_input_vector = vectorizer.transform([user_input]).toarray()
    similarities = cosine_similarity(user_input_vector, vector_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    recommendations = data.iloc[top_indices]
    return recommendations


@app.route('/suggestios/<keyword>', methods=['GET'])
def recommend(keyword):
    recommendations = recommend_opportunities(keyword, volunteering_data, cv, vector_matrix)
    result = recommendations[['opportunity_id', 'title']].to_dict(orient='records')

    return (result)

if __name__ == '__main__':
   app.run(host='127.0.0.1', port=5555, debug=True)


