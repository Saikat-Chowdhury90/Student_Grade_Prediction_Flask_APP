import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
grade = {0: "fair", 1: "good", 2: "poor"}
options = [2, 2, 2, 2, 2, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    arr = np.array([int(x) for x in request.form.values()])
    arr2 = np.array(arr[:15])

    for i, j in zip(options, range(15, 32)):
        temp = np.zeros(i, dtype="int32")
        temp[arr[j]] = 1
        arr2 = np.append(arr2, temp)

    final_features = [np.array(arr2)]
    print(final_features)
    prediction = model.predict(final_features)

    output = grade[prediction[0]]
    print(final_features)
    print(output)

    return render_template('index.html', prediction_text=' Student\'s Final Score should be {} !'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = grade[prediction[0]]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
