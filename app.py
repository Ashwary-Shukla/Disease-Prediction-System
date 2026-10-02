from flask import Flask, render_template, request, session, redirect
import pickle
import numpy as np


app = Flask(__name__)

app.secret_key = "disease_prediction"


import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = pickle.load(
    open(os.path.join(BASE_DIR,"disease_model.pkl"),"rb")
)


symptoms = pickle.load(
    open(os.path.join(BASE_DIR,"symptoms.pkl"),"rb")
)



# Language Dictionary

language_data = {

    "english": {

        "title": "🩺 AI Disease Prediction System",

        "tagline": "Smart Healthcare Solution Powered by Machine Learning",

        "badge": "AI Healthcare Assistant",

        "select": "Select Your Symptoms",

        "button": "Predict Disease",

        "result": "Prediction Result",

        "possible": "Possible Disease"

    },


    "hindi": {

        "title": "🩺 AI रोग पहचान प्रणाली",

        "tagline": "मशीन लर्निंग द्वारा संचालित स्मार्ट स्वास्थ्य समाधान",

        "badge": "AI स्वास्थ्य सहायक",

        "select": "अपने लक्षण चुनें",

        "button": "रोग की जांच करें",

        "result": "जांच परिणाम",

        "possible": "संभावित बीमारी"

    }

}




# Home Page

@app.route("/")
def home():

    lang = session.get("language", "english")


    # refresh hone par prediction remove ho jayega
    prediction = session.pop("prediction", None)



    return render_template(

        "index.html",

        symptoms=symptoms,

        prediction=prediction,

        text=language_data[lang]

    )






# Change Language

@app.route("/language", methods=["POST"])
def change_language():


    lang = request.form["language"]


    session["language"] = lang


    return redirect("/")







# Prediction

@app.route("/predict", methods=["POST"])
def predict():


    selected_symptoms = request.form.getlist("symptoms")



    input_data = np.zeros(len(symptoms))



    for symptom in selected_symptoms:


        if symptom in symptoms:


            index = symptoms.index(symptom)

            input_data[index] = 1





    prediction = model.predict([input_data])[0]
    
    # Prediction temporary store
    session["prediction"] = prediction

    # Redirect to home page
    return redirect("/")

if __name__ == "__main__":

    app.run(debug=True)