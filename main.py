# main.py

import torch
import torch.optim as optim
import torch.nn as nn
from src import config
from src.data_loader import TorchtextDataLoader, get_roberta_dataloaders
from src.model import BiLSTMTagger, FFNNTagger, get_embedding_weights, get_roberta_model
from src.trainer import RnnTrainer, RobertaTrainer
from src.predict import predict_rnn, predict_roberta
from src.utils import count_parameters


def run_rnn_ffnn():
    """Runs the training and evaluation pipeline for BiLSTM or FFNN models."""
    data_handler = TorchtextDataLoader(config)
    text_field, label_field = data_handler.load_and_preprocess()
    train_iter, val_iter, test_iter = data_handler.get_iterators()

    embedding_weights = get_embedding_weights(
        text_field.vocab, config.EMBEDDING_DIM, config.EMBEDDING_TYPE
    )

    model_class = BiLSTMTagger if config.MODEL_TYPE == "BiLSTM" else FFNNTagger
    model = model_class(
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        vocab_size=len(text_field.vocab),
        output_size=len(label_field.vocab),
        embeddings=embedding_weights,
    ).to(config.DEVICE)

    print(
        f"Model: {config.MODEL_TYPE} initialized with {count_parameters(model):,} trainable parameters."
    )

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(
        ignore_index=label_field.vocab.stoi[label_field.pad_token]
    )

    trainer = RnnTrainer(model, optimizer, criterion, label_field, config)
    trainer.train(train_iter, val_iter)

    print("\n--- Final Evaluation on Test Set ---")
    trainer.evaluate(test_iter, phase="Test")


def run_roberta():
    """Runs the fine-tuning and evaluation pipeline for the RoBERTa model."""
    train_dl, val_dl, test_dl, _, label2id, id2label, tokenizer = (
        get_roberta_dataloaders(config)
    )

    model = get_roberta_model(
        config.ROBERTA_MODEL_NAME, label2id, id2label, config.DEVICE
    )
    print(
        f"Model: {config.MODEL_TYPE} initialized with {count_parameters(model):,} trainable parameters."
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    trainer = RobertaTrainer(
        model, optimizer, train_dl, val_dl, test_dl, id2label, config
    )
    trainer.train()

    print("\n--- Final Evaluation on Test Set ---")
    trainer.evaluate(test_dl, "Test")

    # --- Example Prediction ---
    # print("\n--- Example Prediction ---")
    # model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    # model.to(config.DEVICE)
    # test_sentence = "protein D R in RNA Mass-Index"
    # predictions = predict_roberta(test_sentence, model, tokenizer, id2label, config.DEVICE)
    # print(f"Sentence: {test_sentence}")
    # print(f"Predictions: {predictions}")


def main():
    """Main function to select and run the appropriate pipeline."""
    if config.MODEL_TYPE in ["BiLSTM", "FFNN"]:
        run_rnn_ffnn()
    elif config.MODEL_TYPE == "RoBERTa":
        run_roberta()
    else:
        raise ValueError(
            f"Invalid MODEL_TYPE '{config.MODEL_TYPE}' in config.py. Choose from 'BiLSTM', 'FFNN', 'RoBERTa'."
        )


if __name__ == "__main__":
    main()
