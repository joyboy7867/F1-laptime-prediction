import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)

Elastic_model=pickle.load(open('models/F1Elastic.pkl',"rb"))
scaler=pickle.load(open('models/F1Scaler.pkl',"rb"))




@app.route("/",methods=["GET","POST"])
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
        return render_template("result.html", result=main_time)


    else:
        return render_template("home.html")



if __name__ =="__main__":
    app.run(host="0.0.0.0")