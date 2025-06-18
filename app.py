import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)

Elastic_model=pickle.load(open('models/F1Elastic.pkl',"rb"))
scaler=pickle.load(open('models/F1Scaler.pkl',"rb"))
pitscaler=pickle.load(open('models/Pitscaler.pkl','rb'))
pitmodel=pickle.load(open('models/PitModel.pkl','rb'))

@app.route("/",methods=["GET"])
def render():
    return render_template("home.html")

@app.route("/predictLaptime",methods=["GET","POST"])
def predicted_data():
    if request.method=="POST":
        LapNumber=float(request.form.get("LapNumber"))
        Stint=float(request.form.get("Stint"))
        SpeedI2=float(request.form.get("SpeedI2"))
        SpeedFL=float(request.form.get("SpeedFL"))
        SpeedST=float(request.form.get("SpeedST"))
        Compound=float(request.form.get("Compound"))
        TyreLife=float(request.form.get("TyreLife"))
        FreshTyre=float(request.form.get("FreshTyre"))
        TrackStatus=float(request.form.get("TrackStatus"))
        IsAccurate=float(request.form.get("IsAccurate"))
        Team=float(request.form.get("Team"))
        Driver=float(request.form.get("Driver"))
        Position=float(request.form.get("Position"))

        data=[[LapNumber, Stint, SpeedI2, SpeedFL, SpeedST,
                                 Compound, TyreLife, FreshTyre, TrackStatus,Team,
                                 IsAccurate, Driver, Position]]
        scaled_data=scaler.transform(data)
        predicted=Elastic_model.predict(scaled_data)
        corr=pd.to_timedelta(predicted[0],unit='s')
        adjusted_time = corr - pd.to_timedelta("00:00:10")
        main_time=str(adjusted_time).split(" ")[-1]
        return render_template("laptimeresult.html", result=main_time)


    else:
        return render_template("laptimeform.html")

@app.route("/PitPrediction",methods=["GET","POST"])
def pitprediction():
    if request.method=="POST":
        LapNumber=float(request.form.get("LapNumber"))
        Race=float(request.form.get("Race"))
        LapTime=float(request.form.get("LapTime"))
        Compound=float(request.form.get("Compound"))
        Tyreage=float(request.form.get("Tyreage"))
        
        TrackStatus=float(request.form.get("TrackStatus"))
        
       
        Driver=float(request.form.get("Driver"))
        Position=float(request.form.get("Position"))
        sample_pit10 = pd.DataFrame([{
                        'race': Race,
                        'driver': Driver,
                        'lap_number': LapNumber,
                        'compound': Compound,
                        'tyre_age': Tyreage,
                        'track_status': TrackStatus,
                        'position': Position,
                        'lap_time': LapTime,
                                }])         
        
        
        scaled_data1=pitscaler.transform(sample_pit10)
        predicted1=pitmodel.predict(scaled_data1)
        if predicted1[0]== 1:
            result="Pit in this lap"
        else:
            result="Not in this lap"    
        return render_template("pitresult.html", result=result)
    else:
        return render_template("pitform.html")





if __name__ =="__main__":
    app.run(host="0.0.0.0")