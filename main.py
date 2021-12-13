# use naive split strategy to split the whole corpus into smaller pieces (of 10 sentences)
# or run truncated backpropagation through time:
# update weights every k1 steps by running BPTT for k2 steps back
# for instance, in default untruncated version, k1 = 1 and k2 goes all the way to step=t1
# (weights are updated on every step from t1 to T, and to compute the update for the step t,
# all previous step calculations are considered
# some users suggest to use detach on outputs h of LSTM layer
import nltk
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter


def get_paragraphs_for(category=None):
    return nltk.corpus.brown.paras(categories=category)


def paragraph_to_tokens(paragraph, allowed='!?.,:;-+#$%'):
    return [token for sentence in paragraph for token in sentence
            if token.isalpha() or token in allowed]


class Paragraph:
    def __init__(self, paragraph):
        self.paragraph = paragraph

    @property
    def tokens(self):
        pass

    @property
    def embeddings(self):
        pass

    @property
    def one_hot(self):
        pass


class ParagraphsDataset(Dataset):
    start_token = '<Start>'
    end_token = '<End>'

    def __init__(self, paragraphs):
        self.paras = [paragraph_to_tokens(para) for para in paragraphs]

    @property
    def words(self):
        return (word for paragraph in self.paras for word in paragraph)

    def __getitem__(self, index):
        tokens = self.paras[index]
        xs = [self.start_token] + tokens[:-1]
        ys = tokens[1:] + [self.end_token]
        return xs, ys

    def __len__(self):
        return len(self.paras)


def pad(seq, size):
    seq = list(seq)
    while len(seq) < size:
        seq.append(None)
    return seq


def collate(batch):
    inputs, outputs = batch
    longest = max(inputs, key=len)
    max_length = len(longest)

    inputs = [pad(seq, max_length) for seq in inputs]
    outputs = [pad(seq, max_length) for seq in outputs]
    return inputs, outputs


class Encoder:
    oov = '<unknown>'

    def __init__(self, corpus_tokens):
        self.vocab = Counter(train_paragraphs.words).most_common(max_vocab_size)
        self.vocab = self.vocab + [ParagraphsDataset.start_token, ParagraphsDataset.end_token, self.oov]

        self.mapping = {}
        for i, token in enumerate(self.vocab):
            self.mapping[token] = i

    def encode(self, token):
        return self.mapping.get(token, self.mapping[self.oov])

    def decode(self, label):
        if 0 <= label < len(self.vocab):
            return self.vocab[label]
        return self.oov


def to_tensor(encoder, seqs):
    for seq in seqs:
        for token in seq:
            encoder.encode(token)


train_fraction = 0.9
batch_size = 20
genre = 'fiction'
max_vocab_size = 20000

all_paragraphs = get_paragraphs_for(genre)
train_size = len(all_paragraphs) * train_fraction
train_paragraphs = all_paragraphs[:train_size]
test_paragraphs = all_paragraphs[train_size:]

train_dataset = ParagraphsDataset(train_paragraphs)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)

test_dataset = ParagraphsDataset(test_paragraphs)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

vocab = Counter(train_paragraphs.words).most_common(max_vocab_size)


for inputs, outputs in train_dataloader:
    pass


# todo: refactor, separate data preparation code for ease of testing (4 layers of representation: paragraphs, tokens, numbers, vectors)
# todo: cleaning pipeline (extract tokens from raw user entered text, remove forbidden chars, prepend with <start> token)
# todo: apply pipeline to nltk brown dataset (when getting tokens from paragraph)
# todo: build a vocabulary and encoder from cleaned tokens of paragraphs (encoder should encode token to label number and decode back)
# todo: a function to build a tensor from labels lists
# todo: pad using <end> token
# todo: put everything together, create LSTM model, training loop and metrics (accuracy, perplexity)
