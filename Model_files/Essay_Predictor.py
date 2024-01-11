
from functions_for_project import df_maker, make_matrix, model, vec 

from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('testing_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    essay_text = request.form.get('essay_text', ' ')
    if not essay_text.strip():
        return redirect(url_for('index'))
    df = df_maker(essay_text)
    matrix = make_matrix(df, vec)
    prediction = model.predict(matrix)[0]
    if prediction:
        prediction = "AI Generated"
    else:
        prediction = 'Human Written'
    return render_template('testing_result.html', prediction=prediction)

@app.route('/return', methods = ['POST'])
def go_back():
    return redirect(url_for('index'))

def run_flask_app():
    app.run(debug=True)

if __name__ == '__main__':
    run_flask_app()