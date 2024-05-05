from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer
from models.model_utils import ModelUtility
from models.process_data import TextProcessor
from models.data_utils import TextDataset

app = Flask(__name__)

model_path = 'GPT2-kagglewiki-nometa-finetune.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)

model_utility = ModelUtility(model, device)
text_processor = TextProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'text' in request.form:
        text = request.form['text']
    elif 'file' in request.files:
        file = request.files['file']
        text = file.read().decode('utf-8')
    else:
        return render_template('index.html', error="Please provide text or a text file for prediction.")

    normalized_text = text_processor.normalize_text(text)
    cleaned_text = text_processor.remove_stopwords(normalized_text)
    test_loader = text_processor.prepare_data_loader(cleaned_text)

    predicted_probs, predicted_labels, _ = model_utility.model_predict(test_loader)
    prediction = predicted_probs[0]

    pred = 'AI-Generated' if predicted_labels == 1 else 'Human-Written'
    prob = f'{prediction:.6f}'
    
    return render_template('prediction.html', prediction=pred, probability=prob)

if __name__ == '__main__':
    app.run(debug=True)
