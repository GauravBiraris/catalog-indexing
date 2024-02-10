from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Load pretrained BERT 
bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare data
def load_data(file_name):
   return pd.read_csv(file_name)

train_data = load_data('train.csv') 
dev_data = load_data('dev.csv')

# Concatenate title and description 
train_data['text'] = train_data['title'] + ' ' + train_data['description']

# Convert text Series to list of strings
text_data = train_data['text'].tolist()

# Get text column as Series
texts = train_data['text']

# Convert Series to list
text_list = train_data.text.apply(lambda x: [x]).explode()

# Check if this is now a list
print(type(text_list)) 

# Should print <class 'list'>
# Tokenize on new text column
encoded_data = tokenizer(text_data[0])

# Extract input ids and mask                        
input_ids = encoded_data['input_ids']

attention_mask = encoded_data['attention_mask']

# Encode the label strings to numeric values
le = LabelEncoder()
train_labels = le.fit_transform(train_data.label)

# Add classification layer 
num_labels = len(set(train_data.label))
classifier = torch.nn.Linear(bert.config.hidden_size, num_labels)

# Define model, loss, optimizer

class BertForClassification(nn.Module):

  def __init__(self, bert, classifier):
    # initialization code
        super().__init__()
        self.bert = bert 
        self.classifier = classifier
  def forward(self, input_ids, attention_mask, labels=None):
    
    # Pass input to BERT model
    outputs = self.bert(input_ids, attention_mask=attention_mask)
    
    # Take the last hidden state 
    last_hidden_state = outputs[0] 

    # Pass to classifier 
    logits = self.classifier(last_hidden_state)

    # Calculate loss if labels provided
    if labels is not None:
      loss_func = nn.CrossEntropyLoss()
      loss = loss_func(logits, labels)
      return loss
    
    return logits


model = BertForClassification(bert, classifier) 
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-5) 

# Training loop
# Make batches returns tensors
def make_batches(input_ids, attention_mask, labels, batch_size=16):
  # Create tensors 
  input_ids = torch.tensor(input_ids)
  attention_mask = torch.tensor(attention_mask) 
  labels = torch.tensor(labels)
  # Split tensors into batches
  batches = [
    (input_ids[i:i+batch_size],  
     attention_mask[i:i+batch_size],
     labels[i:i+batch_size])
    for i in range(0, len(input_ids), batch_size)
  ]

  return batches

epochs = 3 
batch_size = 16
num_epochs = 3

# Calculate training steps 
num_training_steps = len(train_data) * num_epochs / batch_size
num_warmup_steps = int(0.1*num_training_steps) 

scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps)

# Create batches
batches = make_batches(input_ids, attention_mask, train_labels, batch_size)

for input_ids, attention_mask, labels in batches:
  
  loss = model(input_ids, attention_mask, labels)


for epoch in range(num_epochs):
   for input_ids, attention_mask, labels in batches:
      loss = model(input_ids, attention_mask, labels)
      loss.backward()
      optimizer.step()
      scheduler.step()
      
   # Eval     
   eval(model, dev_data)

# Save fine-tuned model   
model.save('catalog_lm.pt')