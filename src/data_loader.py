# src/data_loader.py

import nltk
import torch
from datasets import load_dataset, DatasetDict
from torchtext.data import Field, Example, Dataset, BucketIterator
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import DataLoader


# --- For RNN/FFNN Models (Torchtext) ---
class TorchtextDataLoader:
    """Handles data loading and preprocessing for RNN/FFNN models using torchtext."""

    def __init__(self, config):
        self.config = config
        self.text_field = Field(
            sequential=True,
            tokenize=lambda x: x,
            include_lengths=True,
            batch_first=True,
        )
        self.label_field = Field(
            sequential=True, tokenize=lambda x: x, is_target=True, batch_first=True
        )
        self.train_data, self.val_data, self.test_data = None, None, None

        try:
            nltk.data.find("tokenizers/punkt")
        except nltk.downloader.DownloadError:
            nltk.download("punkt")

    def _read_data(self, hf_dataset):
        examples = []
        fields = {
            "sentence_tokens": ("text", self.text_field),
            "sentence_labels": ("labels", self.label_field),
        }
        for i in range(len(hf_dataset)):
            tokens = hf_dataset["tokens"][i]
            labels = hf_dataset["ner_tags"][i]
            e = Example.fromdict(
                {"sentence_tokens": tokens, "sentence_labels": labels}, fields
            )
            examples.append(e)
        return Dataset(
            examples, fields=[("text", self.text_field), ("labels", self.label_field)]
        )

    def load_and_preprocess(self):
        dataset = load_dataset(self.config.DATASET_NAME)
        self.train_data = self._read_data(dataset["train"])
        self.val_data = self._read_data(dataset["validation"])
        self.test_data = self._read_data(dataset["test"])
        print("Data loaded and converted to torchtext format.")

        self.text_field.build_vocab(
            self.train_data, max_size=self.config.VOCAB_SIZE_LIMIT
        )
        self.label_field.build_vocab(self.train_data)
        print(
            f"Vocabularies built. Text: {len(self.text_field.vocab)}, Labels: {len(self.label_field.vocab)}"
        )

        return self.text_field, self.label_field

    def get_iterators(self):
        if not self.train_data:
            self.load_and_preprocess()

        train_iter, val_iter, test_iter = BucketIterator.splits(
            (self.train_data, self.val_data, self.test_data),
            batch_size=self.config.BATCH_SIZE,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=self.config.DEVICE,
        )
        return train_iter, val_iter, test_iter


# --- For RoBERTa Model (Transformers) ---
def get_roberta_dataloaders(config):
    """Handles data loading and tokenization for transformer models."""
    dataset = load_dataset(config.DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(config.ROBERTA_MODEL_NAME)

    label_list = dataset["train"].features["ner_tags"].feature.names
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing and aligning labels for RoBERTa...")
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    # Remove original columns to keep the dataset clean
    tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.BATCH_SIZE,
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        collate_fn=data_collator,
        batch_size=config.BATCH_SIZE,
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        collate_fn=data_collator,
        batch_size=config.BATCH_SIZE,
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        label_list,
        label2id,
        id2label,
        tokenizer,
    )
