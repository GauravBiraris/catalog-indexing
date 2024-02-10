import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers import AdamW, get_scheduler


# Load data
df = pd.read_csv('catalog_data.csv')

# Split data
train_df, val_df = train_test_split(df, test_size=0.2) 

# Process train data
# train_texts = train_df['text'].tolist()
# train_labels = train_df['label'].tolist()

# # Process validation data
# val_texts = val_df['text'].tolist() 
# val_labels = val_df['label'].tolist()

# Encode labels and convert to lists
le = LabelEncoder()
train_labels = le.fit_transform(df['label']).tolist()
val_labels = le.transform(val_df['label']).tolist()

# Convert texts to lists 
train_texts = df['description'].tolist()  
val_texts = val_df['description'].tolist()

# Tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Convert to tensors
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)

val_dataset = TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']), 
    torch.tensor(val_labels)
)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16) 

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(le.classes_)
)   

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5) 

num_epochs = 3

# Scheduler 
scheduler = get_scheduler(
    "linear", 
    optimizer=optimizer,
    num_warmup_steps=0, 
    num_training_steps=len(train_loader) * num_epochs
)

# Evaluation function 
def evaluate(model, val_loader):
    model.eval()
    
    total_loss = 0
    
    for batch in val_loader:
        b_input_ids, b_attn_mask, b_labels = batch
        
        with torch.no_grad():        
            loss = model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            
        total_loss += loss.item()
        
    return total_loss / len(val_loader)

# Training loop
num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    
    for batch in train_loader:
        # Unpack batch
        b_input_ids, b_attn_mask, b_labels = batch
        
        # Forward pass 
        loss = model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    val_loss = evaluate(model, val_loader)
    
# Save model
model.save_pretrained('catalog_lm')