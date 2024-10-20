from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

app = Flask(__name__)

# Global variables for the logistic model and dataset
logistic_model = None
logistic_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logistic_regression', methods=['GET', 'POST'])
def logistic_regression():
    global logistic_model, logistic_data
    accuracy = None
    try:
        if request.method == 'POST':
            file = request.files['dataset']
            logistic_data = pd.read_csv(file)
            # Assuming the last column is the target (binary classification)
            X = logistic_data.iloc[:, :-1]
            y = logistic_data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            logistic_model = LogisticRegression()
            logistic_model.fit(X_train, y_train)
            predictions = logistic_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            confusion_mat = confusion_matrix(y_test, predictions)
            accuracy_matrix = pd.DataFrame({
                'Actual': y_test,
                'Predicted': predictions
            })
            return render_template('logistic_regression.html', accuracy=accuracy, confusion_mat=confusion_mat, accuracy_matrix=accuracy_matrix.to_html(classes='table table-striped'))
        return render_template('logistic_regression.html', accuracy=accuracy)
    except Exception as e:
        app.logger.error("Error in /logistic_regression route: %s", e)
        return str(e), 500

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    global logistic_model
    if logistic_model is not None:
        input_data = request.form.to_dict()
        input_data = pd.DataFrame([input_data], dtype=float)
        prediction = logistic_model.predict(input_data)
        return render_template('logistic_regression.html', prediction=prediction[0], accuracy_matrix=None)
    return redirect(url_for('logistic_regression'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
