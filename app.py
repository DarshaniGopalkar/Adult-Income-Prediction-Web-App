import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Set paths based on current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Load the trained Decision Tree model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Helper function to predict income
def ValuePredictor(features_list):
    try:
        features = np.array(features_list).reshape(1, -1)
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        # Collect form values as integers
        to_predict = [
            int(request.form["age"]),
            int(request.form["w_class"]),
            int(request.form["edu"]),
            int(request.form["martial_stat"]),
            int(request.form["occup"]),
            int(request.form["relation"]),
            int(request.form["race"]),
            int(request.form["gender"]),
            int(request.form["c_gain"]),
            int(request.form["c_loss"]),
            int(request.form["hours_per_week"]),
            int(request.form["native-country"])
        ]

        prediction = ValuePredictor(to_predict)

        # Convert prediction to readable text
        if str(prediction) == "1":
            final_prediction = "Income more than 50K"
        else:
            final_prediction = "Income less than or equal 50K"

        return render_template("result.html", prediction=final_prediction)
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
