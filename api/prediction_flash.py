## prediction_flash.py
from flask import Flask,request,jsonify
import glob
from joblib import load

app = Flask(__name__)
best_model = load(glob.glob(".\models\svm_*.joblib")[0])

@app.route("/predict",methods=['POST'])
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
	