"""
Định nghĩa kiến trúc model — copy nguyên từ notebook training.
QUAN TRỌNG: kiến trúc ở đây phải KHỚP HOÀN TOÀN với khi train,
nếu không load_state_dict sẽ báo missing/unexpected keys.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


# ===== Hyperparameters (phải khớp với lúc train) =====
EMBED_SIZE = 256
HIDDEN_SIZE = 512
ATTENTION_DIM = 512


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features = densenet.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, images):
        features = self.features(images)        # [B, 1024, 7, 7]
        features = self.pool(features)          # [B, 1024, 7, 7]

        batch_size = features.size(0)
        features = features.view(batch_size, 1024, -1)   # [B, 1024, 49]
        features = features.permute(0, 2, 1)             # [B, 49, 1024]
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_dim=1024, pad_idx=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(encoder_dim, hidden_size, ATTENTION_DIM)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)
        vocab_size = self.fc.out_features

        embeddings = self.embedding(captions)

        h = torch.zeros(batch_size, self.hidden_size, device=encoder_out.device)
        c = torch.zeros(batch_size, self.hidden_size, device=encoder_out.device)

        outputs = torch.zeros(batch_size, seq_len, vocab_size, device=encoder_out.device)

        for t in range(seq_len):
            context, alpha = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            preds = self.fc(self.dropout(h))
            outputs[:, t] = preds

        return outputs
