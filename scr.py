import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define aspect and sentiment mappings
aspect2idx = {
    'CAMERA': 0, 'FEATURES': 1, 'BATTERY': 2, 'PERFORMANCE': 3,
    'DESIGN': 4, 'GENERAL': 5, 'PRICE': 6, 'SCREEN': 7, 'SER&ACC': 8, 'STORAGE': 9
}
sentiment2idx = {
    'Positive': 2, 'Neutral': 1, 'Negative': 0
}
num_aspect = len(aspect2idx)

idx2aspect = dict(zip(aspect2idx.values(), aspect2idx.keys()))
idx2sentiment = dict(zip(sentiment2idx.values(),sentiment2idx.keys()))

# # Convert label cell to tensor
# def convert_label(cell):
#     return torch.tensor([float(x) for x in cell.strip('[]').split()])

# # Load train data
# train = pd.read_csv("Train_preprocessed_with_-1.csv")
# sentences_train = list(train['comment'])
# labels_train = list(train['label'].apply(convert_label))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')


class AttentionInHtt(nn.Module):
    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)  # (batch_size, seq_len, out_features)
        u = torch.tanh(u)
        similarities = self.uw(u)  # (batch_size, seq_len, 1)
        similarities = similarities.squeeze(dim=-1)  # (batch_size, seq_len)

        # Mask the similarities
        similarities = similarities.masked_fill(~mask.bool(), -float('inf'))

        if self.softmax:
            alpha = torch.softmax(similarities, dim=-1)
            return alpha
        else:
            return similarities
            # return attention score

def element_wise_mul(input1, input2, return_not_sum_result=False):
        output = input1 * input2.unsqueeze(2)  # Ensure correct broadcasting
        result = output.sum(dim=1)
        if return_not_sum_result:
            return result, output
        else:
            return result

class Cae(nn.Module):
    def __init__(self, word_embedder, categories, polarities):
        super().__init__()
        self.word_embedder = word_embedder
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

        embed_dim = word_embedder.embedding_dim
        self.embedding_layer_fc = nn.Linear(embed_dim, embed_dim)
        self.embedding_layer_aspect_attentions = nn.ModuleList([AttentionInHtt(embed_dim, embed_dim) for _ in range(self.category_num)])
        self.lstm_layer_aspect_attentions = nn.ModuleList([AttentionInHtt(embed_dim, embed_dim) for _ in range(self.category_num)])

        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)

        self.category_fcs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(self.category_num)])
        self.sentiment_fc = nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, self.polarity_num))

    def forward(self, tokens, labels, mask, threshold=0.25):
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_sentiment_outputs = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)

            category_output = element_wise_mul(embeddings, alpha)
            embedding_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), word_embeddings).squeeze(1)
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            category_output = element_wise_mul(lstm_result, alpha)
            lstm_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), lstm_result).squeeze(1)
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]], dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        loss = 0
        if labels is not None:
            category_labels = labels[:, :self.category_num]
            polarity_labels = labels[:, self.category_num:]

            for i in range(self.category_num):
                category_mask = (category_labels[:, i] != -1)  # Mask out ignored labels
                sentiment_mask = (polarity_labels[:, i] != -1)

                if category_mask.any():  # Only calculate if there are valid labels
                    category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(-1)[category_mask], category_labels[:, i][category_mask])
                    loss += category_temp_loss

                if sentiment_mask.any():  # Only calculate if there are valid labels
                    sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i][sentiment_mask], polarity_labels[:, i][sentiment_mask].long())
                    loss += sentiment_temp_loss

#         output = {
#             'pred_category': [torch.sigmoid(e) for e in final_category_outputs],
#             'pred_sentiment': [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
#         }
        # formatting output
        final_category_outputs = [torch.sigmoid(e) for e in final_category_outputs]
        final_sentiment_outputs = [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        final_sentiment_outputs = [torch.argmax(e, dim=-1) for e in final_sentiment_outputs]
        
        final_categories = []
        final_sentiments = []

        for i in range(len(final_category_outputs)):
            batch_category = []
            batch_sentiment = []
            for j, category_score in enumerate(final_category_outputs[i]):
                # Apply threshold for aspect detection
                if category_score >= threshold:
                    batch_category.append(1)  # Aspect detected
                    batch_sentiment.append(final_sentiment_outputs[i][j].item())
                else:
                    batch_category.append(0)  # Aspect not detected
                    batch_sentiment.append(-1)  # Set sentiment to -1 for undetected aspect
            final_categories.append(batch_category)
            final_sentiments.append(batch_sentiment)
        final_categories = torch.tensor(final_categories)
        final_sentiments = torch.tensor(final_sentiments)
        
        output = {
            'pred_category': torch.transpose(final_categories, 0, 1), # batch_size*10
            'pred_sentiment': torch.transpose(final_sentiments, 0, 1) # batch_size*10
        }

        return output, loss

def infer_single_comment(model, comment):
    model.eval()
    # Tokenize input text
    encoding = tokenizer(comment, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
#     # Prepare mask for polarity prediction
#     batch_size = 1  # Single inference
#     polarity_mask = torch.ones(batch_size, num_aspect).float()

    # No labels provided in inference
    with torch.no_grad():
        output,loss = model(input_ids, labels=None, mask=attention_mask)

    pred_category = output['pred_category']
    pred_sentiment = output['pred_sentiment']
#     # Extract category and sentiment predictions
#     pred_category = [torch.sigmoid(e).item() for e in output['pred_category']]
#     pred_sentiment = [torch.argmax(e).item() for e in output['pred_sentiment']]

    # Map indices to actual labels
    # aspect_labels = list(aspect2idx.keys())
    # sentiment_labels = list(sentiment2idx.keys())
    # results = {
    #     "Aspect": [aspect_labels[i] for i, val in enumerate(pred_category) if val >= 0.5],
    #     "Sentiment": [sentiment_labels[s] for s in pred_sentiment if s > 0]
    # }
    res = ''
    pred_category = pred_category.squeeze()
    pred_sentiment = pred_sentiment.squeeze()
    for i, v in enumerate(pred_category):
        if v!=0:
            res += f'{idx2aspect[i]}: {idx2sentiment[int(pred_sentiment[i])]}'
            res += '\n'

    return res


w2v = 'W2V_150.txt'
embedding_dim = 150
word_to_vec = {}
with open(w2v, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word_to_vec[word] = vector

vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
E = np.zeros((vocab_size, embedding_dim))
for word, idx in vocab.items():
    E[idx] = word_to_vec.get(word, np.random.normal(scale=0.6, size=(embedding_dim,)))

embedding_matrix = torch.tensor(E, dtype=torch.float32)
embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

categories = aspect2idx.keys()
polarities = sentiment2idx.keys()
model = Cae(embedding_layer, categories, polarities)

model.load_state_dict(torch.load('CAE_checkpoint41.pth'))
model.eval()

sample_comment = "điện thoại pin tốt,nhưng giá cả rất đắt không hợp lý"
x, y = infer_single_comment(model=model, comment=sample_comment)
print("pred_cate", x)
print("pred_sent",y)

