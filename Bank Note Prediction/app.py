from flask import Flask,render_template,request

import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os

# app
app=Flask(__name__)


# Define absolute paths for model and scaler
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "model.h5")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")




# load model and scaler
model=load_model(model_path)
scaler=pickle.load(open(scaler_path,'rb'))


def make_prediction(input_data):
    input_data_scaled = scaler.transform(input_data)
    predictions = model.predict(input_data_scaled)
    predicted_class = (predictions > 0.5).astype(int)


    return predicted_class
    # if predicted_class[0] == 1:
    #     return "Real Note"
    # else:
    #     return "Fake Note"

#routes
@app.route("/")
def index():
    return render_template("index.html")

# routes api end point 

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method=='POST':
        #get
        
        VWTI = float(request.form['VWTI'])
        SWTI = float(request.form['SWTI'])
        CWTI = float(request.form['CWTI'])
        EI = float(request.form['EI'])

        #prepare input data for prediction 
        input_data=np.array([[VWTI,SWTI,CWTI,EI]])

        # Get predictions
        result=make_prediction(input_data)

        if result==1:
            output='real'
        else:
            output='fake'
         
        return render_template('index.html', prediction=output)
    return render_template('index.html',prediction=None)
    
# python main
if __name__ == "__main__":
    app.run(debug=True)