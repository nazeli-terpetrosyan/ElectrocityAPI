from tokenize import Double
import uvicorn
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

app_desc = """<h2>Electrocity API</h2>"""

app = FastAPI(title='Electrocity API', description=app_desc)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('lstm.json', 'r') as json_file:
    json_savedModel= json_file.read()
model = tf.keras.models.model_from_json(json)
model.load_weights('./weights.h5')
scaler = joblib.load("./scaler.save")

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs") 

@app.post("/predict")
async def predict(day1: float, day2: float, day3: float, day4: float, day5: float, day6: float, day7: float, max_temp: float=16, humidity: float=160, wind_speed: float = 30):
    input = np.array([[1., 0., day1, day2, day3, day4, day5, day6, day7]])
    input = scaler.fit_transform(input.reshape(9,-1))
    input = input.reshape( -1, 9)

    pred = model.predict(input[:, np.newaxis, :])
    pred = scaler.inverse_transform(pred)

    return pred[0][0]


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
