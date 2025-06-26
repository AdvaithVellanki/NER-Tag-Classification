# src/predict.py

import torch


# --- Prediction for RNN/FFNN models ---
def predict_rnn(tokenized_sentence, model, text_field, label_field, device):
    model.eval()
    if isinstance(tokenized_sentence, str):
        tokenized_sentence = tokenized_sentence.split()

    numericalized = [
        text_field.vocab.stoi.get(t, text_field.vocab.stoi[text_field.unk_token])
        for t in tokenized_sentence
    ]
    tensor = torch.LongTensor(numericalized).unsqueeze(0).to(device)
    lengths = torch.tensor([len(numericalized)]).cpu()

    with torch.no_grad():
        predictions = model(tensor, lengths).argmax(dim=2).squeeze(0)

    predicted_labels = [label_field.vocab.itos[i] for i in predictions.cpu().numpy()]
    return list(zip(tokenized_sentence, predicted_labels))


# --- Prediction for RoBERTa models ---
def predict_roberta(tokenized_sentence, model, tokenizer, id2label, device):
    model.eval()
    if isinstance(tokenized_sentence, str):
        tokenized_sentence = tokenized_sentence.split()

    inputs = tokenizer(
        tokenized_sentence, return_tensors="pt", is_split_into_words=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [id2label[t.item()] for t in predictions[0]]

    # Align predictions with original words, ignoring special tokens
    word_ids = inputs.word_ids()
    previous_word_idx = None
    aligned_predictions = []
    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        aligned_predictions.append(
            (tokenized_sentence[word_idx], predicted_token_class[i])
        )
        previous_word_idx = word_idx

    return aligned_predictions
