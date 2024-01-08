from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import requests
import numpy as np
from PIL import Image
import os
from time import time
from tqdm import tqdm
import numpy

import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import transforms

from torcheval.metrics.functional import multiclass_f1_score

print('CUDA', torch.cuda.is_available())
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformer
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = torch.load('model.pkl')
model.eval()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'tmp/'


@app.route('/')
def upload_f():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']

        img = Image.open(f.stream)
        img_tensor = tfm(img)
        img_tensor = img_tensor[np.newaxis, :]
        img_tensor = img_tensor.to(device)
        pred_prob = model(img_tensor)
        pred = torch.max(pred_prob, 1).indices
        pred = pred.item()
        softPred = torch.softmax(pred_prob[0], dim=0)[pred]

        if pred == 0:
            return f"Model prediction {softPred} that it is KERATIT"
        elif pred == 1:
            return f"Model prediction {softPred} that it is NORMAL"
        elif pred == 2:
            return f"Model prediction {softPred} that it is TIRED"

        return 'ERROR can not get imagey'


if __name__ == '__main__':
    app.run(debug=True)
