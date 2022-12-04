from fastapi import FastAPI, File, UploadFile
import pickle
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi.responses import StreamingResponse
import io
import datetime


app = FastAPI()


class Item(BaseModel):
    name: str
    year: str
    selling_price: str
    km_driven: str
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: str



@app.post("/predict_item")
def predict_item(item: Item) -> float:
    dataframe1 = pd.DataFrame(columns=['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats'])
    dataframe1.loc[0] = [item.name, item.year, item.selling_price, item.km_driven, item.fuel, item.seller_type, item.transmission, item.owner, item.mileage, item.engine, item.max_power, item.torque, item.seats]
    X_test_cat = dataframe1.drop(["selling_price", "name", "torque"], axis=1)
    y_test_cat = dataframe1['selling_price']
    X_test_cat['mileage'] = X_test_cat['mileage'].astype(str).str.replace(' kmpl', '')
    X_test_cat.loc[X_test_cat['mileage'].str.contains(' km/kg', case=False), 'mileage'] = round(X_test_cat.loc[X_test_cat['mileage'].str.contains(' km/kg', case=False), 'mileage'].astype(str).str.replace('[^0-9.]', "").astype(float)*1.4, 2)
    X_test_cat['mileage'] = X_test_cat['mileage'].astype(float)
    X_test_cat['engine'] = X_test_cat['engine'].astype(str).str.replace(' CC', '').astype(float)
    null_cells1 = X_test_cat['max_power'].isnull()
    X_test_cat['max_power'] = X_test_cat['max_power'].astype(str).mask(null_cells1, np.NaN).str.replace('[^0-9.]', '')
    X_test_cat['max_power'] = pd.to_numeric(X_test_cat['max_power'], errors='coerce')
    X_test_cat['max_power'] = X_test_cat['max_power'].astype(float)
    X_test_cat[['mileage', 'engine', 'max_power', 'seats']] = X_test_cat[['mileage', 'engine', 'max_power', 'seats']].fillna(X_test_cat[['mileage', 'engine', 'max_power', 'seats']].median())
    X_test_cat['seats'] = X_test_cat['seats'].astype(int)
    X_test_cat['engine'] = X_test_cat['engine'].astype(int)
    scaler = StandardScaler()
    scaler.fit(X_test_cat[['km_driven','mileage', 'engine', 'max_power']])
    X_test_cat[['km_driven','mileage', 'engine', 'max_power']] = scaler.transform(X_test_cat[['km_driven','mileage', 'engine', 'max_power']])
    X_test_cat.drop(['km_driven', 'mileage'], axis=1)

    car_old = datetime.datetime.now()
    X_test_cat['horsetimesvolume'] = X_test_cat['engine'] * X_test_cat['max_power']
    X_test_cat['year'] = X_test_cat['year'].apply(lambda x : car_old.year - x)
    X_dum_test = pd.get_dummies(X_test_cat, drop_first=True)


    filename = 'finalized_model.sav'
    print(X_dum_test)
    loaded_model = pickle.load(open(filename, 'rb'))
    pred_mse_elasticnet = loaded_model.predict(X_dum_test)
    return pred_mse_elasticnet


@app.post("/predict_items")
def predict_items(csv_file: UploadFile = File(...)):
    dataframe = pd.read_csv(csv_file.file)

    X_test_cat = dataframe.drop(["selling_price", "name", "torque"], axis=1)
    y_test_cat = dataframe['selling_price']
    X_test_cat['mileage'] = X_test_cat['mileage'].astype(str).str.replace(' kmpl', '')
    X_test_cat.loc[X_test_cat['mileage'].str.contains(' km/kg', case=False), 'mileage'] = round(X_test_cat.loc[X_test_cat['mileage'].str.contains(' km/kg', case=False), 'mileage'].astype(str).str.replace('[^0-9.]', "").astype(float)*1.4, 2)
    X_test_cat['mileage'] = X_test_cat['mileage'].astype(float)
    X_test_cat['engine'] = X_test_cat['engine'].astype(str).str.replace(' CC', '').astype(float)
    null_cells1 = X_test_cat['max_power'].isnull()
    X_test_cat['max_power'] = X_test_cat['max_power'].astype(str).mask(null_cells1, np.NaN).str.replace('[^0-9.]', '')
    X_test_cat['max_power'] = pd.to_numeric(X_test_cat['max_power'], errors='coerce')
    X_test_cat['max_power'] = X_test_cat['max_power'].astype(float)
    X_test_cat[['mileage', 'engine', 'max_power', 'seats']] = X_test_cat[['mileage', 'engine', 'max_power', 'seats']].fillna(X_test_cat[['mileage', 'engine', 'max_power', 'seats']].median())
    X_test_cat['seats'] = X_test_cat['seats'].astype(int)
    X_test_cat['engine'] = X_test_cat['engine'].astype(int)
    scaler = StandardScaler()
    scaler.fit(X_test_cat[['km_driven','mileage', 'engine', 'max_power']])
    X_test_cat[['km_driven','mileage', 'engine', 'max_power']] = scaler.transform(X_test_cat[['km_driven','mileage', 'engine', 'max_power']])
    X_test_cat.drop(['km_driven', 'mileage'], axis=1)

    car_old = datetime.datetime.now()
    X_test_cat['horsetimesvolume'] = X_test_cat['engine'] * X_test_cat['max_power']
    X_test_cat['year'] = X_test_cat['year'].apply(lambda x : car_old.year - x)
    X_dum_test = pd.get_dummies(X_test_cat, drop_first=True)


    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    pred_mse_elasticnet = loaded_model.predict(X_dum_test)
    predicted_price = pd.DataFrame(data=pred_mse_elasticnet)
    predicted_price.columns = ['predicted price']
    result_data = dataframe.join(predicted_price)
    result = loaded_model.score(X_dum_test, y_test_cat)
    # do something with dataframe here (?)
    stream = io.StringIO()
    result_data.to_csv(stream, index = False)
    response = StreamingResponse(iter([stream.getvalue()]),
                            media_type="text/csv")
    
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"

    return response