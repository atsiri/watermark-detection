from flask import (
    Flask, 
    redirect, 
    render_template, 
    request, 
    url_for
)
#from flask_restful import Api
import os
import predict
from PIL import Image
import pickle
#from torchvision import models, transforms
import sys
sys.path.append('../')
from watermarkmodel.model.predictor import WatermarksPredictor
from watermarkmodel.model import get_convnext_model

app = Flask(__name__)
#api = Api(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file
            #file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))
            file.save(os.path.join(basedir,  file.filename)) #app.config['UPLOAD_FOLDER'],
            #file.save(file_path)
            # Redirect to the prediction page
            return redirect(url_for('predict', filename=file.filename))
    return render_template('index.html')

@app.route('/predict/<path:filename>')
def predict(filename): #config
    ##loading the model from the saved file
    pkl_filename = "watermark_model.pkl"
    with open(pkl_filename, 'rb') as f_in:
        modelpkl = pickle.load(f_in)

    transforms = get_convnext_model('convnext-tiny')[1]
    predictor = WatermarksPredictor(modelpkl, transforms, 'cpu')
    prediction = predictor.predict_image(Image.open(filename))

    if prediction == 1:
        result = 'watermark'
    else:
        result = 'non_watermark'
    
    return result
    #use this output for web rendered app
    #return render_template('results.html', prediction=result, filename=filename)

#api.add_resource(predict, '/<path:filename>')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)