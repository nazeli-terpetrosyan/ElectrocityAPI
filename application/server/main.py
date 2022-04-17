from tokenize import Double
import uvicorn
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import numpy as np

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

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs") 

@app.post("/predict")
async def predict(day1: float, day2: float, day3: float, day4: float, day5: float, day6: float, day7: float, max_temp: float=16, humidity: float=160, wind_speed: float = 30):
    mean = np.mean(day1, day2, day3, day4, day5, day6, day7)
    mean -= max_temp/30
    mean -= wind_speed/150
    mean += humidity/150
    return mean

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
