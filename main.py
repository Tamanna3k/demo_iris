from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

ml=joblib.load("C:\demo_iris\demo_iris.pkl")

spec_cls=["setosa","versicolo","virginica"]
class iris_inp(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app=FastAPI()

@app.get('/')
def read_load():
    return "Hi guyss.. Welcome to my Page"

@app.post('/prd')
def prediction(data:iris_inp):
    fea=np.array([[data.sepal_length,data.sepal_width,data.petal_length,data.petal_width]])
    prd_out=ml.predict(fea)
    spe_name=spec_cls[int(prd_out[0])]
    return {"Predicted_species":spe_name}
