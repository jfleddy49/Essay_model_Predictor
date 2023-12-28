
import pickle
from functions_for_project import pattern, add_feat, add_metrics, remove_abstract, get_unique_info, df_maker, make_matrix

from flask import Flask, request, render_template
import os
import webbrowser
from threading import Timer
import threading
import time
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'Classifier_essay.pkl')
vectorizer_path = os.path.join(script_dir, 'Vectorizer_final_essay.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(vectorizer_path, 'rb') as vec_file:
    vec = pickle.load(vec_file)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'Classifier.pkl')

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    essay_text = request.form['essay_text']
    df = df_maker(essay_text)
    matrix = make_matrix(df, vec)
    # Use your machine learning model to make predictions
    prediction = model.predict(matrix)[0]
    if prediction:
        prediction = "AI Generated"
    else:
        prediction = 'Human Written'
    return render_template('result.html', prediction=prediction)


def run_flask_app():
    app.run(debug=True, extra_files=['static/styles.css'], use_reloader=True, port=5000, use_evalex=True, use_debugger=True)

if __name__ == '__main__':
    port = 5000
    url = f'http://127.0.0.1:{port}/'
    run_flask_app()
    webbrowser.open(url=url)