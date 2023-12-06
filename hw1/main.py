from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI()
model = joblib.load('linear_model.pkl')

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return model.predict(item)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return