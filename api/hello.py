from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
best_model = "svm_gamma=0.0005_C=7.joblib"
model = load(best_model)

def predict():
    content = request.json
    img1 = content['image1']
    img2 = content['image2']
    predicted_digit_1 = best_model.predict([img1])
    predicted_digit_2 = best_model.predict([img2])
    if predicted_digit_1 == predicted_digit_2:
        is_same = True
    else:
        is_same = False
    return jsonify({"predicted_digit_1":str(predicted_digit_1[0]),
                    "predicted_digit_2":str(predicted_digit_2[0]),
                    "is_image_same":is_same})

if __name__ == "__main__":
    app.run(port=5000)
