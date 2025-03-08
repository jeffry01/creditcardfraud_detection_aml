from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__, template_folder="Templates/")

model = joblib.load(open("Models/credit_card_fraud_model.pkl", "rb"))
scaler = joblib.load(open("Models/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    print('data')
    data = np.array(list(data.values())).reshape(1, -1)
    scaled_data = scaler.transform(data)
    output = model.predict(scaled_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("index.html", prediction_text="Ouput of fraud detection is {}".format(int(output)))

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render (or default to 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
