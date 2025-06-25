# src/config.py

import torch

# -- Training Hyperparameters --
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 10

# -- Model Configuration --
MODEL_TYPE = "BiLSTM"  # Options: 'BiLSTM', 'FFNN'
EMBEDDING_TYPE = "Word2Vec"  # Options: 'Word2Vec', 'GloVe'

# -- Model Dimensions --
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
VOCAB_SIZE_LIMIT = 20000

# -- Dataset and Paths --
DATASET_NAME = "surrey-nlp/PLOD-CW"
MODEL_SAVE_PATH = "model.pt"

# -- Device Configuration --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
