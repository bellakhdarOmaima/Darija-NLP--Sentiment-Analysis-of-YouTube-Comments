import torch
import torch.nn as nn
import pickle
from preprocess import preprocess_text

MODEL_PATH = "models/bilstm_cnn_model.pth"
VOCAB_PATH = "vocab/vocablstm.pkl"
LABEL_ENCODER_PATH = "vocab/label_encoderlstm.pkl"

with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

class BiLSTM_CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, kernel_sizes, num_filters):
        super(BiLSTM_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, 2 * hidden_size)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_size]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, 2 * hidden_size]
        lstm_out = lstm_out.unsqueeze(1)  # Add channel dimension: [batch_size, 1, seq_len, 2 * hidden_size]
        conv_outs = [torch.relu(conv(lstm_out)).squeeze(3) for conv in self.convs]  # Apply each Conv2D
        conv_outs = [torch.max(pool, 2)[0] for pool in conv_outs]  # Max pooling
        x = torch.cat(conv_outs, 1)  # Concatenate feature maps
        x = self.dropout(x)
        x = self.fc(x)
        return x

embed_size = 128
hidden_size = 64
num_classes = len(label_encoder.classes_)
kernel_sizes = [3, 4, 5]
num_filters = 100
vocab_size = len(vocab) + 1

model = BiLSTM_CNN(vocab_size, embed_size, hidden_size, num_classes, kernel_sizes, num_filters)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def tokens_to_indices(tokens, vocab, max_len=60):
    indices = [vocab.get(token, 0) for token in tokens[:max_len]]  # Truncate ou remplace par 0 si le token est absent
    return indices + [0] * (max_len - len(indices))  # Padding pour atteindre max_len


def lstm_cnn_predict(sentence):
    indices = tokens_to_indices(preprocess_text(sentence), vocab)
    input_tensor = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([predicted_class])[0]
