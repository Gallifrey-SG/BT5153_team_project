import torch
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import pandas as pd
import yaml

# Load configuration from the yaml file
with open('DistilBERT_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_config = config['model']
train_config = config['training']
dataset_config = config['dataset']

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained(model_config['model_name_or_path'])
model = DistilBertForSequenceClassification.from_pretrained(model_config['model_name_or_path'], num_labels=2)
model.to(device)

def encode_texts(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=model_config['max_length'], return_tensors="pt")


df = pd.read_csv(train_config['dataset_path'])
texts = df[dataset_config['text_column']].tolist()
labels = df[dataset_config['label_column']].tolist()

inputs = encode_texts(texts)
inputs['labels'] = torch.tensor(labels)

dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])


train_size = int((1 - train_config['validation_split']) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

optimizer = AdamW(model.parameters(), lr=train_config['learning_rate'])

model.train()
for epoch in range(train_config['num_epochs']):
    total_loss = 0
    for batch in train_loader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{train_config['num_epochs']}, Loss: {total_loss / len(train_loader)}")

model.save_pretrained(train_config['save_path'])
tokenizer.save_pretrained(train_config['save_path'])

print("Training complete.")
