from flask import Flask, request, render_template
import numpy as np
import joblib

# Load trained model
obj = joblib.load('california.joblib')

model = obj['Model']
columns = obj['Columns']

app = Flask(__name__)


# GET method
# Used to display the home page
@app.route('/', methods=['GET'])
def home():

    return render_template(
        'index.html',
        columns=columns
    )


# POST method
# Used to receive input and predict
@app.route('/predict', methods=['POST'])
def predict():

    Input = []

    for column in columns:

        value = request.form.get(column)

        value = float(value)

        Input.append(value)

    # Convert input into numpy array
    Input = np.array(Input).reshape(1, -1)

    # Prediction
    prediction = model.predict(Input)[0]

    return render_template(
        'index.html',
        columns=columns,
        prediction=prediction
    )


if __name__ == '__main__':
    app.run(debug=True)