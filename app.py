import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data = pd.read_csv("mod.csv")
ohe = pd.read_csv("ohe.csv")
ohe.drop(['Unnamed: 0'],axis = 1, inplace = True)
pipe = pickle.load(open("BangaloreHousePrice.pkl","rb"))

@app.route("/")
def index():

    locations = sorted(data["location"].unique())
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    
    if request.method == "POST":
        
        lis = list(ohe.columns)
        for index, item in enumerate(lis):
            if item == request.form["location"]:
                lis[index] = 1
            else:
                lis[index] = 0
        
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        bhk = int(request.form['BHK'])
        
        temp_array = [total_sqft, bath, balcony, bhk] + lis
        
        data = np.array([temp_array])
        prediction = pipe.predict(data)
        
        

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Approx Price for your search >-~ Rs {} Lakhs'.format(output))
        
                
        
        


if __name__ == "__main__":
    app.run(debug = True)
