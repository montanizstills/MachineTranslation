from collections import Counter
from torchtext.data.utils import get_tokenizer
import torch
from torch import nn
from torch.nn import functional as F


# Exercise 1: Build custom tokenizer and graph results against well known tokenizers, such as spacy, bpe (mini), sentencepiece, tiktoken, spacy, etc.
# Exercise 2: Implement a transformer model with one head of self attention
# Exercise 3: Implement a transformer model with multiple heads of self attention
# Exercise 4: Implement a transformer model with multiple heads of self attention and feed forward neural network
# Exercise 5: Implement a transformer model with multiple heads of self attention, feed forward neural network, and positional encoding
# Exercise 6: Implement a transformer model with multiple heads of self attention, feed forward neural network, positional encoding, and layer normalization
# Exercise 7: Implement a transformer model with multiple heads of self attention, feed forward neural network, positional encoding, layer normalization, and residual connections
# Plot Transformer training loss and accuracy with a plot function in the Transform model itself. Visit Karpathy video 3



def load_data(file, encoding='utf-8'):
    with open(file, mode='r', encoding=encoding) as f:
        lang1 = [line.split('\t')[0] for line in f.readlines()]
        lang2 = [line.split('\t')[1] for line in f.readlines()]
    return lang1, lang2


class Tokenize:
    def __init__(self, language):
        self.language = language
        self.tokenizer = get_tokenizer('spacy', language=self.language)
        self.vocab = None

    def tokenize_sentence(self, sentence):
        return self.tokenizer(sentence)

    def train(self, corpus):
        word_freq = Counter()
        _ = [word_freq.update(self.tokenize_sentence(sentence)) for sentence in corpus]
        self.vocab = {word: i for i, (word, freq) in enumerate(word_freq.items())}  # enumerate list and unpack tuple


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        # Query, key, and value are all linear layers.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # create a tril matrix of ones
        # PyTorch naming convention because the tril is not a parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)  # (B, T, head_size)
        query = self.query(x)  # (B, T, head_size)
        value = self.value(x)  # (B, T, head_size)

        # Dot product the key and the query to get the weights
        w = key @ query.transpose(-2, -1)  # (B,T,H) @ (B,H,T) = (B, T, T)

        # Dividing by sqrt(head_size) for stability and making sure the variance stays close to zero
        w = w * (C ** -0.5)

        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        out = w @ value  # (B, T, T) @ (B, T, C) = (B, T, C) cuz B stays the same so essentially its a (T, T) @ (T, C)
        return out



class Transformer:
    def __init__(self, language):
        self.language = language
        self.tokenizer = Tokenize(language)
        self.tokenizer.train(english_corpus)
        self.MAX_LEN = 512

    def encode(self, sentence):
        tokens = self.tokenizer.tokenize_sentence(sentence)
        return [self.tokenizer.vocab[token] for token in tokens

    def add_positional_encoding(self, sentence):

        # Initialize a matrix (max_seq_len, embed_dim) for positional encodings
        emb: torch.Tensor
        pos_enc = torch.zeros((self.MAX_LEN, emb.shape[-1])) # emb.shape should actually be embed_dim

        # Compute the positional encodings
        for pos in range(self.MAX_LEN):
            for i in range(0, emb.shape[-1], 2):
                pos_enc[pos, i] = torch.sin(pos / (10000 ** ((2 * i) / emb.shape[-1])))
                if i + 1 < emb.shape:
                    pos_enc[pos, i + 1] = torch.cos(pos / (10000 ** ((2 * (i + 1)) / emb.shape[-1])))

        return torch.from_numpy(pos_enc).float()

    def train(self):
        pass

    # call forward pass of transformer to predict next token with one head of self attention
    def predict_next_token(self, sentence):
        head = Head(64, 512, 10)
        encoded_sentence = self.encode(sentence)
        # tokenize input sentence and return its tensor
        tensor_sentence = torch.tensor(encoded_sentence).unsqueeze(0)  # convert to tensor and add batch dimension
        return head(tensor_sentence)


if __name__ == "__main__":
    english_corpus, french_corpus = load_data('eng-fra.txt')

    print(english_corpus[:5])
    # print(french_corpus)

    # t = Tokenize('en')  # remove redundancy of creating Tokenize object, called in tokenize_sentence
    # t.train(english_corpus)
    # print(t.vocab)
    # tokens = t.tokenize_sentence(french_corpus[0])
    # print(tokens)
