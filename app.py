from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
stream_encoder = joblib.load("stream_encoder.pkl")
career_encoder = joblib.load("career_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    physics = float(request.form["Physics"])
    chemistry = float(request.form["Chemistry"])
    maths = float(request.form["Mathematics"])
    biology = float(request.form["Biology"])
    english = float(request.form["English"])
    computer = float(request.form["Computer_Science"])
    economics = float(request.form["Economics"])
    accountancy = float(request.form["Accountancy"])
    business = float(request.form["Business_Studies"])
    history = float(request.form["History"])
    geography = float(request.form["Geography"])
    political = float(request.form["Political_Science"])
    percentage = float(request.form["Aggregate_Percentage"])

    stream = request.form["Stream"]
    stream = stream_encoder.transform([stream])[0]

    new_data = pd.DataFrame([{
        "Physics": physics,
        "Chemistry": chemistry,
        "Mathematics": maths,
        "Biology": biology,
        "English": english,
        "Computer_Science": computer,
        "Economics": economics,
        "Accountancy": accountancy,
        "Business_Studies": business,
        "History": history,
        "Geography": geography,
        "Political_Science": political,
        "Stream": stream,
        "Aggregate_Percentage": percentage
    }])

    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    predicted_career = career_encoder.inverse_transform(prediction)

    return render_template(
        "index.html",
        prediction=predicted_career[0]
    )

if __name__ == "__main__":
    app.run(debug=True)
    
