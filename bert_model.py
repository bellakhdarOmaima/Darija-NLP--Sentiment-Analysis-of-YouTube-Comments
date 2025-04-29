import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

MAX_LEN = 256

# Classe BertClassifier
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 3

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


# Préparer les données pour BERT
def preprocessing_for_bert(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

        if len(encoded_sent['input_ids']) > 1:
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# Prédiction avec BERT
def bert_predict(model, sentences):
    input_ids, attention_masks = preprocessing_for_bert(sentences)

    dataset = TensorDataset(input_ids, attention_masks)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    model.eval()
    all_logits = []

    for batch in dataloader:
        b_input_ids, b_attn_mask = tuple(t for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

# Charger le modèle
model = BertClassifier(freeze_bert=False)
model.load_state_dict(torch.load('models/bert_classifier_model.pt', map_location=torch.device('cpu')))
model.eval()

# Mapper les labels
label_map = {0: 'neutral', 1: 'negative', 2: 'positive'}