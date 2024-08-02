pip install Flask pandas scikit-learn joblib

#Example Application Code:

from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Example model training
def train_model(data):
    # Assume the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'model.pkl')

    # Predict on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    data = pd.read_csv(file)
    accuracy = train_model(data)
    return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)














