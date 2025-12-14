from enum import Enum

from fastapi import FastAPI, Query, HTTPException
import pickle
import pandas as pd
import uvicorn
from pydantic import BaseModel
from starlette import status
from starlette.middleware.cors import CORSMiddleware

from preprocess_catboost import Preprocess_catboost
import __main__
__main__.Preprocess_catboost = Preprocess_catboost
# --- Загружаем модель ---
with open("catboost_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# --- Создаём приложение ---
app = FastAPI(title="Insurance Cross Selling Prediction")


class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

    def get_value(self):
        return int(self.value == self.MALE)


class VehicleAgeEnum(str, Enum):
    LESS_THAN_ONE = "< 1 Year"
    FROM_1_TO_2 = "1-2 Year"
    GREATER_THAN_2 = "> 2 Years"


class VehicleDamageEnum(str, Enum):
    YES = "Yes"
    NO = "No"

    def get_value(self):
        return int(self.value == self.YES)


class ResultSchema(BaseModel):
    prediction: int
    probability: float


# --- GET эндпоинт ---
@app.get("/predict")
def predict(
        Gender: GenderEnum = Query(...),
        Age: int = Query(...),
        Driving_License: int = Query(...),
        Region_Code: int = Query(...),
        Previously_Insured: int = Query(...),
        Vehicle_Age: VehicleAgeEnum = Query(...),
        Vehicle_Damage: VehicleDamageEnum = Query(...),
        Annual_Premium: int = Query(...),
        Policy_Sales_Channel: int = Query(...),
        Vintage: int = Query(...),
):
    if Region_Code < 0 or Region_Code > 52:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect Region_Code")

    if Age < 0 or Age > 85:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect Age, maximum age=85")

    if Policy_Sales_Channel < 1 or Policy_Sales_Channel > 163:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect Policy_Sales_Channel")

    if Driving_License not in [0, 1]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect Driving_License")

    if Previously_Insured not in [0, 1]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect Previously_Insured")

    if Annual_Premium < 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect Annual_Premium")

    if Vintage < 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect Vintage")

    value = {
        'Gender': Gender.get_value(),
        'Age': Age,
        'Driving_License': Driving_License,
        'Region_Code': Region_Code,
        'Previously_Insured': Previously_Insured,
        'Vehicle_Age': Vehicle_Age.value,
        'Vehicle_Damage': Vehicle_Damage.get_value(),
        'Annual_Premium': Annual_Premium,
        'Policy_Sales_Channel': Policy_Sales_Channel,
        'Vintage': Vintage,
    }
    # Формируем DataFrame для модели
    data = pd.DataFrame([value])

    # Получаем предсказание
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]  # вероятность положительного класса

    return ResultSchema(
        prediction=int(pred),
        probability=float(prob)
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r'^http://178\.72\.151\.49(:[0-9]+)?$',  # все порты
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Запуск локально ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
