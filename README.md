# watermark-detection

Fine-tuned watermark detection model for tiny-sized dataset, using pretrained ConvNEXT-tiny architecture.

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

* Download the model file from [here](https://huggingface.co/atsiri/convnext_watermark-detection)

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
access the app.py
```bash
cd flask-api
python app.py
```

call API from python
```bash
import requests

filename = 'wm1.jpg'
urlimage = 'http://127.0.0.1:5000/predict/' + filename
response = requests.get(url)
print(response.text)
```

### Kubernetes Deployment


## Train Model
### Dataset Preprocessing
```bash
from watermarkmodel.model.dataset import WatermarkDataset
import pandas as pd

#normalization
preprocess = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RandomRotation([90, -90], 0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#read dataset
df_train = pd.read_csv('../dataset/train_data_v1.csv')
df_val = pd.read_csv('../dataset/val_data_v1.csv')

#create dataset
train_ds = WatermarkDataset(df_train, preprocess['train'])
val_ds = WatermarkDataset(df_val, preprocess['val'])
datasets = {
    'train': train_ds,
    'val': val_ds,
}
```

### Train Model
```bash
import warnings
warnings.filterwarnings("ignore")
from watermarkmodel.model.convnext import convnext_tiny
from watermarkmodel.model.train import train_model

#load model
model_ft = convnext_tiny(pretrained=True, in_22k=True, num_classes=21841)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#config
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(params=model_ft.parameters(), lr=0.2e-5)
BATCH_SIZE = 8
dataloaders_dict = {
    x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #to prevent runtimeerror on non gpu device
    for x in ['train', 'val']
}

#NN model set up
model_ft.head = nn.Sequential( 
    nn.Linear(in_features=768, out_features=512),
    nn.GELU(),
    nn.Linear(in_features=512, out_features=256),
    nn.GELU(),
    nn.Linear(in_features=256, out_features=2),
)

#train
model_ft, train_acc_history, val_acc_history = train_model(
    model_ft, dataloaders_dict, criterion, optimizer, num_epochs=10
)
```

### Model Evaluation
Evaluate test images
```bash
import warnings
warnings.filterwarnings("ignore")
from watermarkmodel.utils import list_images
from watermarkmodel.model import get_convnext_model
from watermarkmodel.model.predictor import WatermarksPredictor
import pickle

#validation data
images = list_images('../images/test_images/') 

pkl_filename = "watermark_model.pkl"
with open(pkl_filename, 'rb') as f_in:
    modelpkl = pickle.load(f_in)

transforms = get_convnext_model('convnext-tiny')[1]
predictor = WatermarksPredictor(modelpkl, transforms, 'cpu')
result = predictor.run(images)
```

Plot Confusion Matrix
```bash
```
