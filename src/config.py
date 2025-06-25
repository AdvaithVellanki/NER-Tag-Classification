# src/config.py

import torch
import torch.nn as nn

# -- General Configuration --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "surrey-nlp/PLOD-CW"

# -- Model Selection --
# Choose the model to run. Options: 'BiLSTM', 'FFNN', 'RoBERTa'
MODEL_TYPE = "RoBERTa"

# -- Common Hyperparameters --
LEARNING_RATE = 2e-5  # A good default for transformers. 1e-3 is good for RNNs.
BATCH_SIZE = 16
NUM_EPOCHS = 3  # Transformers fine-tune faster. Use 10-30 for RNNs.

# -- RNN/FFNN Specific Configuration --
if MODEL_TYPE in ["BiLSTM", "FFNN"]:
    EMBEDDING_TYPE = "Word2Vec"  # Options: 'Word2Vec', 'GloVe'
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    VOCAB_SIZE_LIMIT = 20000
    MODEL_SAVE_PATH = f"{MODEL_TYPE.lower()}_{EMBEDDING_TYPE.lower()}_model.pt"

# -- Transformer Specific Configuration --
else:
    ROBERTA_MODEL_NAME = "surrey-nlp/roberta-large-finetuned-abbr-filtered-plod"
    MODEL_SAVE_PATH = "roberta_finetuned_model.pt"
