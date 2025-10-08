from flask import Flask, render_template, request, jsonify
from backend import predict 
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')

# @app.route('/api/run_model', methods=['POST'])
# def run_model():
#     file = request.files['file']  # get uploaded file
#     file_path = os.path.join('data', file.filename)
#     os.makedirs('data', exist_ok=True)  # create folder if it doesn't exist
#     file.save(file_path)

#     # Run prediction
#     result = predict(file_path)

#     if result == "impairment":
#         return jsonify({"prediction": "match"})
#     else:
#         return jsonify({"prediction": "no_match"}) 
@app.route('/api/run_model', methods=['POST'])
def run_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join('data', file.filename)
    os.makedirs('data', exist_ok=True)
    file.save(file_path)

    result = predict(file_path)

    if result == "impairment":
        return jsonify({"prediction": "match"})
    else:
        return jsonify({"prediction": "no_match"})
