from flask import Flask
from flask import request
from flask import jsonify




app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p><i>Hello, World!</i></p>"


# get x and y somehow
#     - query parameter
#     - get call / methods
#     - post call / methods



app = Flask(__name__)

@app.route("/predict",methods=['POST'])
def predict():
    content = request.json
    img1 = content['image']
    model = content['model_name']
    
    if model=="svm":
        best_model = load("./svm_gamma=0.001_C=0.5.joblib")
    elif model=="tree":
        best_model = load("./tree_max_depth_8_Criterion_entropy")

    predicted_digit = best_model.predict([img1])
    
    return jsonify({"predicted_digit":str(predicted_digit[0]),
                    "model":model})

if __name__ == "_main_":
    app.run(
