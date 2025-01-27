import torch
from torch import nn
import numpy as np

num_tokens = 500
token_dim = 1024
embedding_dim = 128
hidden_dim = 128
nheads = 4
n_layers = 3
dropout = 0.1
n_bytes = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0:: 2] = torch.sin(position * div_term)
        pe[:, 1:: 2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe[None, :, :]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CLTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=token_dim, embedding_dim=embedding_dim)

        # positional encoding layer
        self.pe = PositionalEncoding(d_model=embedding_dim, dropout=dropout, max_len=num_tokens)

        # encoder layers
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nheads, dim_feedforward=hidden_dim,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=n_layers)
        self.last_ln = nn.LayerNorm(embedding_dim)

        # final dense layer
        self.dense = nn.Linear(in_features=embedding_dim, out_features=1)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=n_bytes, stride=n_bytes, padding=0)
        if n_bytes == 1:
            self.conv1d = nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.int32)
        x = self.embedding(x)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.last_ln(x)
        x = self.dense(x).transpose(dim0=1, dim1=2)
        x = self.conv1d(x).squeeze(dim=1)
        x = self.sigmoid(x)

        return x
