# NER Tag Classification with RNNs and FFNNs

This project performs Named Entity Recognition (NER) on the PLOD-CW dataset from the Hugging Face `datasets` library. It implements and compares 3 different neural network architectures: a Bidirectional LSTM (BiLSTM), a Feed-Forward Neural Network (FFNN) and a state of the art **RoBERTa Transformer** model for sequence tagging.

The fine-tuned RoBERTa transformer model achieves an **accuracy of 96%** on the test set.
## Features

- **Data Handling**: Loads and preprocesses the `surrey-nlp/PLOD-CW` dataset.
- **Text Encodings**: Utilizes pre-trained word embeddings, with options for:
  - Word2Vec (Google News 300)
  - GloVe (Wikipedia Gigaword 300)
- **Model Architectures**:
  - A robust **BiLSTM** tagger for capturing sequential context.
  - A baseline **FFNN** tagger.
  - A powerful **RoBERTa** model for fine-tuning (`surrey-nlp/roberta-large-finetuned-abbr-filtered-plod`).
- **Experimentation**: Easily configurable settings for model type, embeddings, learning rate, and epochs via a central `config.py` file.
- **Evaluation**: Calculates and displays key metrics including Accuracy, Precision, Recall, F1-Score, and a confusion matrix.
- **Inference**: Provides a simple interface to run predictions on new, tokenized text.

## Project Structure

```
/NER_BiLSTM_Project/
|
|-- Notebooks/
|   |-- NER.ipynb             # Jupyter Notebook file containing the entire code
|-- src/
|   |-- config.py             # All hyperparameters and settings
|   |-- data_loader.py        # Dataset loading and preprocessing
|   |-- model.py              # Model architectures (BiLSTMTagger, FFNNTagger, RoBERTa)
|   |-- trainer.py            # Training and evaluation logic
|   |-- predict.py            # Inference function
|   |-- utils.py              # Helper functions
|
|-- main.py                   # Main script to run the project
|-- requirements.txt          # Dependencies
|-- README.md                 # This documentation
```

## Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/AdvaithVellanki/NER-Tag-Classification.git>
    cd NER-Tag-Classification
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    The first time you run the script, it will download the necessary `punkt` tokenizer from NLTK.

### Configuration

All major parameters can be adjusted in `src/config.py`. This includes:
- MODEL_TYPE = 'RoBERTa'  # Options: 'BiLSTM', 'FFNN', 'RoBERTa'
  - For `BiLSTM` or `FFNN`, you can also set `EMBEDDING_TYPE`.
  - For `RoBERTa`, the script will use the model specified in `ROBERTA_MODEL_NAME`.
- `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`.
- `EMBEDDING_DIM`, `HIDDEN_DIM`.

### Training and Evaluation

Simply run the main script. It will automatically use the settings from `config.py` to load the appropriate data pipeline and model, then proceed with training and evaluation.

```bash
python main.py
```

The script will download the required datasets and pre-trained models from `gensim` or `transformers` on its first run, which may take some time.

### Making Predictions

Example prediction logic is included in `main.py` (commented out by default). You can uncomment it and provide a sentence to see the model's predictions. The script will automatically use the correct prediction function based on the trained model type.

Example sentence for prediction:
```python
sentence = ["protein", "D", "R", "in", "RNA", "Mass-Index"]
```