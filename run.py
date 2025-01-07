import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('hw2_train.csv')

x = df['utterances']
y = df['IOB Slot tags']

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize utterances
tokenized_utterances = [tokenizer.encode(utt, truncation=True, padding="max_length", max_length=50) for utt in x]

label_encoder = LabelEncoder() # Label encoding for IOB Slot tags
# Flatten tags for fitting the encoder, then reshape to original form after encoding
y_flattened = [tag for tags in y for tag in tags.split()]
label_encoder.fit(y_flattened)
np.save('label_classes.npy', label_encoder.classes_)

encoded_tags = [label_encoder.transform(tags.split()) for tags in y]

# Padding encoded tags to ensure equal lengths (max length = 50)
max_len = 50
encoded_tags_padded = [np.pad(tag, (0, max_len - len(tag)), constant_values=0) for tag in encoded_tags]

bert_model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=len(label_encoder.classes_)
)

class POSDataset(Dataset):
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        # Convert lists to PyTorch tensors
        text_tensor = torch.tensor(text, dtype=torch.long)
        tags_tensor = torch.tensor(tags, dtype=torch.long)
        return text_tensor, tags_tensor

#Data split, 80% of data for training and 20% for testing
train_texts, val_texts, train_tags, val_tags = train_test_split(tokenized_utterances, encoded_tags_padded, test_size=0.2, random_state=42)

#Initialize training and testing datasets and loaders
train_dataset = POSDataset(train_texts, train_tags)
val_dataset = POSDataset(val_texts, val_tags)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#Model architecture
class SeqTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=110, num_filters=127, kernel_size=3, hidden_dim=296):
        super().__init__()
        self.bert = bert_model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1) # CNN Layer
        self.gru = nn.GRU(num_filters, hidden_dim, batch_first=True, bidirectional=True) # GRU Layer
        self.fc = nn.Linear(hidden_dim * 2, tagset_size) #Bidirectionality
        self.dropout = nn.Dropout(0.49201211676346956) # Dropout for regularization

    def forward(self, sentences, tags=None):
        # Embedding layer
        embedded = self.embedding(sentences) # (batch_size, seq_len, embedding_dim)
        
        # Apply Conv1D layer
        embedded = embedded.permute(0, 2, 1) # (batch_size, embedding_dim, seq_len) for Conv1D
        conv_out = self.conv1d(embedded) # (batch_size, num_filters, seq_len)
        conv_out = torch.relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1) # (batch_size, seq_len, num_filters)

        gru_out, _ = self.gru(conv_out) # (batch_size, seq_len, hidden_dim)

        # Apply dropout and fully connected layer
        gru_out = self.dropout(gru_out)
        emissions = self.fc(gru_out)               # (batch_size, seq_len, tagset_size)
        return emissions

EMBEDDING_DIM = 110
HIDDEN_DIM = 296
BATCH_SIZE = 32
LEARNING_RATE = 0.0009725782027345584
NUM_EPOCHS = 50

vocab_size = tokenizer.vocab_size
tagset_size = len(label_encoder.classes_)

model = SeqTagger(vocab_size, tagset_size, EMBEDDING_DIM, HIDDEN_DIM)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_losses = []
val_losses = []
f1_scores = []

# Move model to device
model = model.to(device)

# Training loop
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Reshape outputs and labels for loss calculation
        outputs = outputs.view(-1, tagset_size)
        labels = labels.view(-1)

        # Calculate loss and update weights
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Optional: Print batch loss for monitoring
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Calculate loss
            outputs_reshaped = outputs.view(-1, tagset_size)
            labels_reshaped = labels.view(-1)
            val_loss = loss_fn(outputs_reshaped, labels_reshaped)
            total_val_loss += val_loss.item()

            # Get predictions and accumulate for F1 score
            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())

    # Filter padding tokens (assuming padding index is 0)
    filtered_preds = [p for p, l in zip(all_preds, all_labels) if l != 0]
    filtered_labels = [l for l in all_labels if l != 0]

    # Compute train and validation loss
    train_loss = total_train_loss / len(train_loader)
    val_loss = total_val_loss / len(val_loader)

    # Calculate F1 score
    f1 = f1_score(filtered_labels, filtered_preds, average='weighted')

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    f1_scores.append(f1)

    print(f'Epoch = {epoch + 1} | Train Loss = {train_loss:.3f} | Val Loss = {val_loss:.3f} | F1 Score = {f1:.3f}')


# Load the test data
df_test = pd.read_csv('hw2_test.csv')
test_ids = df_test['ID']

# Tokenize test utterances and get actual lengths
tokenized_test_utterances = [
    tokenizer.encode(utt, truncation=True, padding="max_length", max_length=50)
    for utt in df_test['utterances']
]
actual_lengths = [len(tokenizer.tokenize(utt)) for utt in df_test['utterances']]
test_inputs = torch.tensor(tokenized_test_utterances, dtype=torch.long).to(device)

# Generate predictions
model.eval()
predictions = []
with torch.no_grad():
    for i in range(0, len(test_inputs), BATCH_SIZE):
        batch_outputs = model(test_inputs[i:i + BATCH_SIZE])
        batch_preds = torch.argmax(batch_outputs, dim=2).cpu().numpy()

        # Decode and trim predictions
        predictions.extend([
            " ".join(label_encoder.inverse_transform(pred[:actual_lengths[i + idx]]))
            for idx, pred in enumerate(batch_preds)
        ])

# Create submission file
pd.DataFrame({'ID': test_ids, 'IOB Slot tags': predictions}).to_csv('gru_cnn_bert_bi.csv', index=False)
print("Predictions saved in desired format")
