# Importing essential libraries and modules

import pickle
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request
from markupsafe import Markup

from utils.fertilizer import fertilizer_dic
from crop_details import crop_details

import config
from pymongo import MongoClient


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------



# Loading crop recommendation model

crop_recommendation_model_path = 'NBClassifier.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------



app = Flask(__name__)

#Connection to mongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['Agrovision']
collection = db['cropResult']
collection=db['fertilizerResult']


# render home page


@ app.route('/')
def home():
    title = 'AgroVision - Home'
    return render_template('home.html', title=title)

# render crop recommendation form page
@ app.route('/crop')
def crop_recommend():
    title = 'AgroVision - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgroVision- Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render aboutus page
@ app.route('/aboutus')
def aboutus():
    title = 'AgroVision- About US'

    return render_template('aboutus.html', title=title)


# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Agrovision - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            dataMongo = {
                'crop':final_prediction
            }
            collection.insert_one(dataMongo)
            response = Markup(str(crop_details[final_prediction]))
            return render_template('crop-result.html', prediction=final_prediction,crop_recommend=response, title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgroVision - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])
    df = pd.read_csv('fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)
# ===============================================================================================
if __name__ == '__main__':
        app.run(debug=False)