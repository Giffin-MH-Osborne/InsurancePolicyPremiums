from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='pages')
model = load_model('models/K_Means_Model')

cols = []

with open('cols.csv', 'r') as f:
    lines = f.read().replace('\ufeff','')
    cols = lines.split(',')

@app.route('/')
def home():
    return render_template('index.html', pred='', cols=cols)

""" 
Input Array -> Sorted Array
0: Gender -> 2
1: Age -> 3
2: Date of Birth -> 4
3: Age at Eff Date -> 14

4: Area ->  6
5: Traffic Index -> 7
6: Vehicle Age -> 8
7: Vehicle Body -> 9
8: Vehicle Value -> 10

9: Policy Number -> 0
10: Policy Effective Date -> 1

11: Credit Score -> 5

12: Claim Office -> 11
13: Number of Claims -> 12
14: Total Claims ->  13

"""

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    int_features = [x for x in request.form.values()]
    
    personal_data = int_features[:4]
    vehicle_data = int_features[4:9]
    financial_data = int_features[9:]

    sorted = []

    for i in range(len(financial_data)):
        if(i < 2):
            sorted.append(financial_data[i])
        elif(i == 2):
            sorted = sorted + personal_data[:-1]
            sorted.append(financial_data[i])
            sorted = sorted + vehicle_data
        else:
            sorted.append(financial_data[i])
            print(financial_data[i])
    sorted.append(personal_data[-1])

    final = np.array(sorted)

    data_unseen = pd.DataFrame([final], columns= cols)

    prediction=predict_model(model, data=data_unseen, round=0)
    
    print(prediction)
    cluster = int(prediction.Label[0])

    if(cluster==0):
        result = "Low"
    elif(cluster == 1):
        result = "High-Middle"
    elif(cluster == 2):
        result = "Low-Middle"
    else:
        result = "High"


    return render_template('index.html', pred='This customer presents {} Risk'.format(result))