# watermark-detection

## Installation:

```bash
git clone https://github.com/atsiri/watermark-detection
cd watermark-detection
pip install -r requirements.txt
```

Repository contents:
1. [Images](https://github.com/atsiri/watermark-detection/tree/main/images) (contains initial image datasets)
2. [Dataset](https://github.com/atsiri/watermark-detection/tree/main/watermarkmodel) (processed image datasets, *split into train, test, and validation sets*)
3. [Model](https://github.com/atsiri/watermark-detection/tree/main/watermarkmodel) (model functions)
4. [Notebook](https://github.com/atsiri/watermark-detection/tree/main/notebook) (guide to run and train the model)
5. Flask-API (REST-API files)

## Usage
### Basic Usage
#### Model Load
```bash
from PIL import Image
from watermarkmodel.model import get_watermarks_detection_model
from watermarkmodel.model.predictor import WatermarksPredictor

#model load
model, transforms = get_watermarks_detection_model('convnext-tiny', pretrained=True,
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
fp16=False, cache_dir='../watermarkmodel/model/models')
```

#### Detect Watermark of a Single Image
```bash
#detect watermark
predictor = WatermarksPredictor(model, transforms, 'cpu')
prediction = predictor.predict_image(Image.open(images[0]))
print('watermark' if prediction==1 else 'non_watermark')
Image.open(images[0])
```

#### Detect Watermarks of Images in a Folder
```bash
#load images
images = list_images('../images/test_images/')
#run predictor
predictor = WatermarksPredictor(model, transforms, 'cpu')
result = predictor.run(images)
```

### REST-API Usage
```bash
cd flask-api
python app.py
http://127.0.0.1:5000
```
### Kubernetes Deployment


## Train Model

##
