from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # Import CORS


# Loading trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# API endpoint for the chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        # Parse user input from the POST request
        data = request.get_json()
        age = data.get("age")
        gender = data.get("gender")
        purchase_amount = data.get("purchase_amount")

        # Validate input
        if age is None or gender not in ["Male", "Female"] or purchase_amount is None:
            return jsonify({"response": "Please provide valid inputs: age, gender, and purchase_amount."})

        # Convert inputs into numerical format
        gender_numeric = 1 if gender == "Male" else 0
        
        features = np.array([[age, gender_numeric, purchase_amount]])

        # Predict using the ML model
        prediction = int(model.predict(features)[0])
        print(f"Prediction value: {prediction}")  # Debugging
        

        # Customize bot's response based on the prediction and age condition
        if 0 < age < 15:
            response = "This customer is likely to churn due to age-specific conditions."
        elif prediction == 0:
            response = "This customer is likely to be retained."
        else:
            response = "This customer is likely to churn."

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
