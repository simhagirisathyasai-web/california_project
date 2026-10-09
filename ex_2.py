from flask import Flask, request
import joblib

obj = joblib.load('california1.joblib')

model = obj['Model']
columns = obj['Columns']

print("Columns:", columns)

app = Flask(__name__)


@app.route('/')
def main():
    return "Welcome"


@app.route('/predict')
def predict():

    Input = []

    for i in columns:
        val = request.args.get(i)
        Input.append(float(val))

    out = model.predict([Input])

    return str(out[0])


if __name__ == '__main__':
    app.run(debug=True)