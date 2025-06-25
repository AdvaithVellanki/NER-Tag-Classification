# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim.downloader as api
from transformers import AutoModelForTokenClassification


# --- Functions for RNN/FFNN ---
def get_embedding_weights(field_vocab, embedding_dim, embedding_type):
    """Loads pre-trained embeddings and creates a weights matrix."""
    print(f"Loading pre-trained {embedding_type} embeddings...")
    if embedding_type == "Word2Vec":
        model = api.load("word2vec-google-news-300")
    elif embedding_type == "GloVe":
        model = api.load("glove-wiki-gigaword-300")
    else:
        raise ValueError("Unsupported embedding type")

    matrix_len = len(field_vocab)
    weights_matrix = torch.zeros((matrix_len, embedding_dim))

    for i, word in enumerate(field_vocab.itos):
        try:
            weights_matrix[i] = torch.from_numpy(model[word]).clone()
        except KeyError:
            weights_matrix[i] = torch.randn((embedding_dim,))

    print("Embedding weights matrix created.")
    return weights_matrix


class BiLSTMTagger(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, output_size, embeddings=None
    ):
        super().__init__()

        if embeddings is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2 * hidden_dim, output_size)

    def forward(self, text, text_lengths):
        embeddings = self.embeddings(text)
        packed_embeddings = pack_padded_sequence(
            embeddings, text_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_embeddings)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits


class FFNNTagger(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, output_size, embeddings=None
    ):
        super().__init__()
        if embeddings is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, text, text_lengths=None):
        embeddings = self.embeddings(text)
        x = self.dropout(self.relu(self.fc1(embeddings)))
        logits = self.fc2(x)
        return logits


# --- Function for RoBERTa ---
def get_roberta_model(model_name, label2id, id2label, device):
    """Loads a pre-trained RoBERTa model for token classification."""
    print(f"Loading {model_name} from Hugging Face Hub...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    ).to(device)
    return model
