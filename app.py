from flask import Flask, render_template, request
import pickle
import numpy as np
import json
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

models = {
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "KNN": pickle.load(open("models/knn.pkl", "rb")),
    "Naïve Bayes": pickle.load(open("models/naïve_bayes.pkl", "rb")),
}

with open("models/accuracies.json", "r") as f:
    model_accuracies = json.load(f)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
      
        features = [float(request.form[key]) for key in ["nitrogen", "phosphorus", "potassium", 
                                                         "temperature", "humidity", "ph", "rainfall"]]
        logging.debug(f"Received input: {features}")

        input_data = np.array([features])
        
        predictions = {}
        for model_name, model in models.items():
            crop = model.predict(input_data)[0]
            predictions[model_name] = {
                "crop": crop,
                "accuracy": model_accuracies[model_name]["Accuracy"],
                
            }

        return render_template("result.html", predictions=predictions)

    except Exception as e:
        logging.error(f"Error: {e}")
        return render_template("index.html", error="Invalid input! Please enter numeric values.")

if __name__ == "__main__":
    app.run(debug=True)
