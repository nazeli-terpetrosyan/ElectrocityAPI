import uvicorn
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

import numpy as np
import pandas as pd
import tensorflow as tf

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

model = tf.keras.models.load_model('LSTM_model')

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs") 

@app.post("/predict")
async def predict(prev: str, max_temp: int=16, humidity: int=160, wind_speed: int = 30):
    prev = prev.split(",")

    return model.summary()


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
