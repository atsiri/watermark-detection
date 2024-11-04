import torch
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from watermarkmodel.utils import list_images
from watermarkmodel.model import get_convnext_model
from watermarkmodel.model.predictor import WatermarksPredictor
from watermarkmodel.model.metrics import plot_confusion_matrix
import pandas as pd
import pickle
import os

def evaluate_ml_model():
    images = list_images('../images/test_images/') 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pkl_filename = 'watermark_model.pkl'
    with open(pkl_filename, 'rb') as f_in:
        modelpkl = pickle.load(f_in)

    transforms = get_convnext_model('convnext-tiny')[1]
    predictor = WatermarksPredictor(modelpkl, transforms, device)
    result = predictor.run(images)

    df_testcsv = pd.read_csv('../dataset/test_data_v1.csv')
    df_testcsv['filename'] = df_testcsv['path'].apply(os.path.basename)

    df_testresult = pd.DataFrame(list(zip(images, result)), columns=['path', 'prediction'])
    df_testresult['filename'] = df_testresult['path'].apply(os.path.basename)

    df_result = df_testcsv.merge(df_testresult, left_on='filename', right_on='filename')[['filename', 'label', 'prediction']]
    accuracy = df_result[df_result.label == df_result.prediction]['filename'].count() / len(df_result) * 100

    #metric result
    confmatrix = plot_confusion_matrix(df_result['label'].values, df_result['prediction'].values)
    precision = confmatrix[1][1]
    recall = confmatrix[2][1]
    figures = confmatrix[0].figure_.savefig('confusion_matrix.png')

    return precision, recall, accuracy, figures