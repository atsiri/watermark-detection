import pickle
import pandas as pd
import json

def predict(imagefile): #config
    ##loading the model from the saved file
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    predictor = WatermarksPredictor(model, transforms, 'cpu')
    prediction = predictor.predict_image(Image.open(imagefile))

    if prediction == 1:
        return('watermark', Image.open(imagefile))
    else:
        return('non_watermark', Image.open(imagefile))