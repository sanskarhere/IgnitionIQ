from flask import Flask,render_template,request,jsonify,session
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)

#LOGGING
logging.basicConfig(filename='logging.log',level=logging.INFO,format="%(levelname)s-%(asctime)s-%(message)s")

app.secret_key='sgsbnsh'
#LOAD MODELS
model=pickle.load(open('models/model.pkl','rb'))
Standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET']) 
def predict_data():
    logging.info("Render predict Page")
    return render_template('predict.html')
@app.route('/predict_datapoint_',methods=['GET','POST'])
def result():
    Temperature=float(request.form.get('Temperature'))
    RH=float(request.form.get('RH'))
    Ws=float(request.form.get('Ws'))
    Rain=float(request.form.get('Rain'))
    FFMC=float(request.form.get('FFMC'))
    DMC=float(request.form.get('DMC'))
    ISI=float(request.form.get('ISI'))
    Classes=float(request.form.get('classes'))
    Region=float(request.form.get('Region'))
    new_data_scaled=Standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
    try:
       result=model.predict(new_data_scaled)
       
    except Exception as e:
       logging.info(f"Prediction made: Temp={Temperature}, RH={RH},DMC={DMC}")
       logging.error(f"Prediction failed: {str(e)}")
    fwi=result[0]
    logging.info(f"Prediction made: Temp={Temperature}, RH={RH}, ,DMC={DMC},FWI={fwi}")
    return render_template('result.html',fwi=fwi)
print('http://localhost:5000')
app.run(host='0.0.0.0', port=5000,debug=True)
