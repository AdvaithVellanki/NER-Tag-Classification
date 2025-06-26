# src/trainer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall


class RnnTrainer:
    """Trainer for BiLSTM and FFNN models."""

    def __init__(self, model, optimizer, criterion, label_field, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.label_field = label_field
        self.config = config

    def train(self, train_iter, val_iter):
        print("Starting training...")
        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            total_loss = 0

            for batch in train_iter:
                text, text_lengths = batch.text
                labels = batch.labels
                self.optimizer.zero_grad()
                outputs = self.model(text, text_lengths.cpu())
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_iter)
            val_metrics = self.evaluate(val_iter, "Validation")
            print(
                f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")

    def evaluate(self, data_iter, phase="Evaluation"):
        self.model.eval()
        true_labels, pred_labels = [], []

        with torch.no_grad():
            for batch in data_iter:
                text, text_lengths = batch.text
                labels = batch.labels
                outputs = self.model(text, text_lengths.cpu())
                predictions = outputs.argmax(dim=2)
                for i, length in enumerate(text_lengths):
                    true_labels.extend(labels[i][:length].cpu().numpy())
                    pred_labels.extend(predictions[i][:length].cpu().numpy())

        report_labels = [
            i
            for i, label in enumerate(self.label_field.vocab.itos)
            if label not in ["<unk>", "<pad>"]
        ]
        label_names = [self.label_field.vocab.itos[i] for i in report_labels]

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            labels=report_labels,
            average="macro",
            zero_division=0,
        )

        print(f"\n--- {phase} Metrics ---")
        print(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
        )

        if phase == "Test":
            cm = confusion_matrix(
                true_labels, pred_labels, labels=report_labels, normalize="true"
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=label_names
            )
            disp.plot()
            plt.show()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


class RobertaTrainer:
    """Trainer for fine-tuning the RoBERTa model."""

    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        id2label,
        config,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.id2label = id2label
        self.config = config

    def train(self):
        print("Starting RoBERTa fine-tuning...")
        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            total_loss = 0
            for batch in tqdm(
                self.train_dataloader, desc=f"Training Epoch {epoch + 1}"
            ):
                batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)
            val_metrics = self.evaluate(self.val_dataloader, "Validation")

            print(
                f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}"
            )

        torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
        print(f"Fine-tuned RoBERTa model saved to {self.config.MODEL_SAVE_PATH}")

    def evaluate(self, dataloader, phase="Evaluation"):
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in tqdm(dataloader, desc=f"{phase}"):
            batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            predictions = outputs.logits.argmax(dim=2)
            labels = batch["labels"]

            for pred_seq, label_seq in zip(predictions, labels):
                true_pred, true_label = [], []
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        true_pred.append(self.id2label[p.item()])
                        true_label.append(self.id2label[l.item()])
                all_preds.append(true_pred)
                all_labels.append(true_label)

        flat_true = [label for sublist in all_labels for label in sublist]
        flat_pred = [label for sublist in all_preds for label in sublist]

        accuracy = accuracy_score(flat_true, flat_pred)
        f1 = seqeval_f1(all_labels, all_preds)
        precision = seqeval_precision(all_labels, all_preds)
        recall = seqeval_recall(all_labels, all_preds)

        print(f"\n--- {phase} Metrics ---")
        print(
            f"Accuracy: {accuracy:.4f}, Precision (seqeval): {precision:.4f}, Recall (seqeval): {recall:.4f}, F1 Score (seqeval): {f1:.4f}"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
