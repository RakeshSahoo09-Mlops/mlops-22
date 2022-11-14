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



@app.route("/sum", methods=['POST'])
def sum():
    print(request.json)
    x = request.json['x']
    y = request.json['y']
    z= x + y
    return jsonify('y')


    return -1
    # z = x + y
    # return z