import torch
import torch.nn as nn
import pickle
from preprocess import preprocess_text

# Charger les fichiers nécessaires
MODEL_PATH = "models/cnn_sentiment_model.pth"
VOCAB_PATH = "vocab/vocab.pkl"
LABEL_ENCODER_PATH = "vocab/label_encoder.pkl"

with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes, num_filters):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_filters, (k, embed_size)),
                nn.BatchNorm2d(num_filters),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_size]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, seq_len, embed_size]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply each Conv2D
        x = [torch.max(pool, 2)[0] for pool in x]  # Max pooling
        x = torch.cat(x, 1)  # Concatenate feature maps
        x = self.dropout(x)
        x = self.fc(x)
        return x

embed_size = 128
num_classes = len(label_encoder.classes_)
kernel_sizes = [2, 3, 4, 5, 6]
num_filters = 150
vocab_size = len(vocab) + 1

model = CNNModel(vocab_size, embed_size, num_classes, kernel_sizes, num_filters)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def tokens_to_indices(tokens, vocab, max_len=100):
    indices = [vocab.get(token, 0) for token in tokens[:max_len]]
    return indices + [0] * (max_len - len(indices))

def cnn_predict(sentence):
    indices = tokens_to_indices(preprocess_text(sentence), vocab)
    input_tensor = torch.tensor([indices], dtype=torch.long)
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]
