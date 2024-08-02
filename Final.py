from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            df = pd.read_csv(file)
        else:
            data = request.form['data']
            df = pd.read_csv(data)
        
        # Dummy ML model for illustration purposes
        X = df.drop('species', axis=1)
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions)
        
        return render_template('result.html', report=report)

    return '''
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="text" name="data" placeholder="Or paste data here">
            <input type="submit" value="Upload Data">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)