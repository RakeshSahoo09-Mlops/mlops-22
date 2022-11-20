from flask import Flask
from flask import request
from flask import jsonify
from joblib import load



app = Flask(__name__)
model_path = "svm_gamma=0.0005_C=5.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<p><i>Hello, World!</i></p>"



@app.route("/sum", methods=['POST'])
def sum():
    print(request.json)
    x = request.json['x']
    y = request.json['y']
    z= x + y
    return jsonify({'sum': z})



@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}