import pickle
import numpy as np

def load_file():
    with open('lr_model.pkl', 'rb') as lr_file:
        lr_model = pickle.load(lr_file)

    with open('nn_model.pkl', 'rb') as nn_file:
        nn_model = pickle.load(nn_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    with open('encoder.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    return lr_model, nn_model, encoder, scaler

def preprocess(SquareFeet, Bedrooms, Bathrooms, Neighborhood, YearBuilt, encoder, scaler):
    encoded_data = encoder.transform([[Neighborhood]]).toarray() 

    num_data = np.array([[SquareFeet, Bedrooms, Bathrooms, YearBuilt]])
    
    if encoded_data.ndim == 1:
        encoded_data = encoded_data.reshape(1, -1)

    full_data = np.hstack((num_data, encoded_data))

    scaler_data = scaler.transform(full_data)

    return scaler_data

def predict(preprocessed_data, model):
    result = model.predict(preprocessed_data)
    return result[0]


lr_model, nn_model, encoder, scaler = load_file()

preprocessed_data = preprocess(432, 3, 1, "Rural", 1969, encoder, scaler)
print(predict(preprocessed_data, lr_model))

