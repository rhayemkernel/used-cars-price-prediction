import pickle
import numpy as np
import json

__model = None
__columns = None

def load_artifacts():
    global __model
    global __columns
    global __name

    with open('server/artifacts/columns.json','r') as f:
        __columns = json.load(f)['data_columns']
        __name=__columns[:22]
    with open('server/artifacts/car price prediction.pickle','rb') as file:
        __model = pickle.load(file)

def predict_price(name,transmission,fuel,owner,year,km_driven,engine,max_power):
    global __model
    global __columns
    x = []
    x[:26] = np.zeros(32,dtype='int32')
    x[27] = owner
    x[28] = year
    x[29] = km_driven
    x[30] = engine
    x[31] = max_power
    name_index = __columns.index(name.lower())
    transmission_index = __columns.index(transmission.lower())
    fuel_index = __columns.index(fuel.lower())

    
    if name_index>=0:
        x[name_index] = 1
    if transmission_index>=0:
        x[transmission_index] = 1
    if fuel_index>=2:
        x[fuel_index] = 1
        
    return float(format(__model.predict([x])[0],'.2f'))
def get_car_name():
    return __name
if __name__ == '__main__':
    
    load_artifacts()
    print(get_car_name())
    print(predict_price('Tata','Manual','Diesel',1,2013,10000,2050,150))