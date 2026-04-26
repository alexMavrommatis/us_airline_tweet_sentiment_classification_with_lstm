#============================================================================
# An custom python model that provide functionality for preparing
# a dataset of tokens for input in an LSTM model
#============================================================================

import pandas as pd
import numpy as np
import nltk
from nltk.probability import FreqDist

def get_vocabulary(data, min_freq=2):
    """Find and return a dictionary with all the words that
       appear in dataset and have minimum frequency of appearance=min_freq

    Args:
       data(pd.Series): A pd.Series where each row is a list of tokens.
       min_freq(int): The minimum frequency a word appears in the Series(default=2)
    Returns:
        vocabulary(dict): A dictionary with keys all the words and values
        their respective frequency of appearance.
       """
    all_tokens = [token for tweets in data for token in tweets]
    fdist = FreqDist(all_tokens)
    top_words = fdist.most_common()
    freq_df = pd.DataFrame(top_words, columns=['word', 'count'])
    freq_df = freq_df[freq_df['count'] >= min_freq]

    vocabulary = freq_df.set_index('word')['count'].to_dict()

    return vocabulary

def set_words_2ids(vocabulary):
    """Parse a dictionary and assign an index values to each key

    Args:
        vocabulary(dict): A dictionary with word tokens as keys
    Returns:
        word2id(dict): A new dictionary with an index for each value.
    """

    word2id = {"<pad>": 0, "<unk>": 1}
    for i, word in enumerate(vocabulary.keys()):
        word2id[word] = i + 2
    return word2id

# word2id.get('blahblah',1)

def pad_sequences(sequence, word2id, max_words):
    """Encodes a sequence of tokenized texts into integer IDs and pads/truncates
    them to a fixed length.

    Args:
        sequence (Iterable[List[str]]): An iterable (e.g., pandas Series or
            list) where each element is a list of string tokens representing
            a single text sample.
        word2id (Dict[str, int]): Mapping from vocabulary words to integer
            IDs. Must include the special tokens '<pad>' (ID = 0) and
            '<unk>' (ID = 1).
        max_words (int): Target sequence length. Shorter sequences are padded
            with zeros; longer sequences are truncated to this length.

    Returns:
        np.ndarray: A 2D integer array of shape (n_samples, max_words),
            where n_samples is the number of input sequences.
    """

    encoded = [[word2id.get(t, 1) for t in tokens] for tokens in sequence]
    padded = [tokens+([0] * (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in encoded]

    return np.array(padded)


def create_embedding_matrix(glove, word2id, dim=100):
    """Builds an embedding matrix aligned with the vocabulary, using pretrained
      GloVe vectors.

      For each word in `word2id`, this function fills row `id` of the matrix

      Args:
          glove (gensim.models.KeyedVectors or Dict[str, np.ndarray]): Pretrained
          word embeddings supporting `word in glove` and `glove[word]` access.
          word2id (Dict[str, int]): Mapping from vocabulary words to integer IDs.
          Must include '<pad>' at ID = 0 and '<unk>' at ID = 1.
          dim (int): Dimensionality of the embedding vectors. Must match the
          dimensionality of the vectors in `glove`. Default is 100.

      Returns:
          np.ndarray: An embedding matrix of shape (len(word2id), dim), where
              row `i` contains the vector for the word with ID = i.
    """

    embedding_matrix = np.zeros((len(word2id), dim))
    for word, id in word2id.items():
        if word in glove:
            embedding_matrix[id] = glove[word]
        elif id == 0:
            continue
        elif id == 1:
            embedding_matrix[id] = np.random.normal(0.0, 0.1, size= dim)
        else:
            embedding_matrix[id] = np.random.uniform(0.0, 0.1, size= dim)

    return embedding_matrix