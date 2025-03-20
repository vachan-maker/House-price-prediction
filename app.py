from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route (renders form)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route (handles form submission & API requests)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form if request.form else request.get_json()

        # Extract input values
        area = float(data["area"])
        bedrooms = int(data["bedrooms"])
        bathrooms = int(data["bathrooms"])
        stories = int(data["stories"])

        # Prepare input
        features = np.array([[area, bedrooms, bathrooms,stories]])

        # Predict price
        predicted_price = model.predict(features)[0]

        return render_template("result.html", price=round(predicted_price, 2))

    except Exception as e:
        return render_template("result.html", error=str(e))

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
