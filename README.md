# Airline Tweet Sentiment Classification with Bi-LSTM

Classification of US airline tweets into **Negative**, **Neutral**, and **Positive** sentiment using a bi-LSTM neural network built with PyTorch.
This project compares a Bi-LSTM with mean pooling against the same model with self-attention, evaluating accuracy and macro-F1 on a class-imbalanced dataset.

## Dataset

Twitter US Airline Sentiment dataset (~14,640 tweets).

## Pipeline

1. **Preprocessing** Build vocabulary -> Map tokens to IDs -> Pad -> Build embedding matrix
2. **Bi-LSTM - with attention**: frozen embedding → dropout → stacked Bi-LSTM → dropout → attention → linear classifier
3. **Bi-LSTM - no attention**: frozen embedding → dropout → stacked Bi-LSTM → dropout → mean pooling → linear classifier
4. **Train** with Adam + weight decay, gradient clipping, and early stopping on validation loss
5. **Evaluate** on test set with classification report and confusion matrix

## Results

| Model                  | Accuracy | Macro F1 |
|------------------------|----------|----------|
| Bi-LSTM (baseline)     | 0.768    | 0.695    |
| Bi-LSTM + attention    | 0.777    | 0.700    |

## Project Structure

```
├── data/
│   ├── 01_raw/Tweets.csv
│   └── 02_processed/Tweets.csv
├── src/
│   ├── utils.py
│   ├── nn_utils.py
│   └── lstm_classifier.py
├── notebooks/
│   └── main_notebook.ipynb
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch, scikit-learn, NLTK, gensim, pandas, numpy, seaborn, matplotlib

Install with:
```bash
pip install torch scikit-learn nltk gensim pandas numpy seaborn matplotlib
```

## How to Run

Open and run the main notebook end-to-end. Ensure the dataset is placed under `data/02_processed/Tweets.csv`.