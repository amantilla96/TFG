import numpy as np
import pandas as pd

#CARGAMOS LOS DATOS
building = pd.read_csv("../input/ashrae-great-energy-predictor-iii-dataset/building_metadata.csv")
train = pd.read_csv("../input/ashrae-great-energy-predictor-iii-dataset/train.csv")
weather = pd.read_csv("../input/ashrae-great-energy-predictor-iii-dataset/weather_train.csv")

#Corregido error de muchos decimales
train['meter_reading'] = train.meter_reading.round(4)

#Unimos toda la base de datos
train = train.merge (building,on="building_id", how="left")
train = train.merge(weather, on=["site_id", "timestamp"], how="left")

#Dividimos la columna timestamp en day y hour
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["day"] = train["timestamp"].dt.strftime("%Y-%m-%d")
train["hour"] = train["timestamp"].dt.strftime("%H:%M:%S")

#Eliminamos la columna timestamp de Train
train.drop(["timestamp"], axis=1, inplace=True)

#Separamos Train en los 4 dispositivos de medida.
trainM0 = train[train['meter'] == 0] 
trainM1 = train[train['meter'] == 1] 
trainM2 = train[train['meter'] == 2] 
trainM3 = train[train['meter'] == 3] 

# Free memory
del train
del building
del weather

#Guardamos cada conjunto de datos, según los medidores
trainM0.to_csv("trainM0.csv")
trainM1.to_csv("trainM1.csv")
trainM2.to_csv("trainM2.csv")
trainM3.to_csv("trainM3.csv")
