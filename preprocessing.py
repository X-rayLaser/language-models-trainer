from functools import reduce
from operator import add
from collections import Counter

import torch


def tokenize_prompt(prompt, allowed_punctuation='!?.,:;-+#$%'):
    """Turns a raw prompt text into a sequence of tokens.

    Makes all letters lower case.

    Usage:
    >>>tokenize_prompt('Hello world!')
    >>>['Hello', 'world', '!']

    :param prompt: str
    :param allowed_punctuation: an iterable of handled punctuation characters
    :return: list[str]
    """
    texts = prompt.lower().split(' ')
    tokens = [[s] if s.isalpha() else [s[:-1], s[-1]] for s in texts if s]
    initial = []
    tokens = reduce(add, tokens, initial)
    return [t for t in tokens if t.isalpha() or t in allowed_punctuation]


def paragraph_to_tokens(paragraph, allowed='!?.,:;-+#$%'):
    return [token for sentence in paragraph for token in sentence
            if token.isalpha() or token in allowed]


def wrapped_tokens(tokens):
    return [wrapped_tokens.start_token] + tokens + [wrapped_tokens.end_token]


wrapped_tokens.start_token = '<Start>'
wrapped_tokens.end_token = '<End>'


def build_vocab(tokens, max_size):
    if max_size <= 0:
        return []

    counts = Counter(tokens).most_common(max_size)
    return sorted(word for word, _ in counts)


class ExampleFactory:
    def __init__(self, tokens, encoder):
        self.tokens = list(tokens)
        self.encoder = encoder

    def encode(self):
        return [self.encoder.encode(token) for token in self.tokens]

    def make_example(self):
        labels = self.encode()
        xs = labels[:-1]
        ys = labels[1:]
        return xs, ys

    def make_batch(self):
        xs, ys = self.make_example()
        xs = xs.unsqueeze(0)
        ys = ys.unsqueeze(0)
        return xs, ys


class Encoder:
    oov_token = '<OOV>'

    @classmethod
    def deserialize(cls, s):
        return Encoder(s.split(' '))

    @classmethod
    def build(cls, tokens):
        vocab = sorted(set(tokens))
        return Encoder(vocab)

    def __init__(self, vocab):
        self._vocab = list(vocab)
        self._validate_vocab(self._vocab)
        self.word_to_num = {word: i for i, word in enumerate(self._vocab)}

    def __len__(self):
        return len(self.vocab) + 1

    @property
    def vocab(self):
        return self._vocab

    def encode(self, token):
        return self.word_to_num.get(token, len(self.vocab))

    def encode_many(self, tokens):
        return [self.encode(t) for t in tokens]

    def decode(self, number):
        return self.vocab[number] if 0 <= number < len(self.vocab) else self.oov_token

    def decode_many(self, numbers):
        return [self.decode(n) for n in numbers]

    def serialize(self):
        return ' '.join(self.vocab)

    def _validate_vocab(self, vocab):
        with_spaces = [word for word in vocab if ' ' in word]
        if any(with_spaces):
            raise ValueError(f'Vocab words must not contain whitespaces. '
                             f'Whitespace words: {with_spaces}')


class Collator:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pad(self, seq, size, filler=None):
        seq = list(seq)
        while len(seq) < size:
            seq.append(filler)
        return seq

    def __call__(self, batch):
        inputs, targets = batch
        longest = max(inputs, key=len)
        max_length = len(longest)

        filler = targets[0][-1]
        inputs = [self.pad(seq, max_length, filler) for seq in inputs]
        targets = [self.pad(seq, max_length, filler) for seq in targets]

        num_classes = self.num_classes
        return one_hot_tensor(inputs, num_classes), one_hot_tensor(targets, num_classes)


def one_hot_tensor(classes, num_classes):
    """Form a 1-hot tensor from a list of class sequences

    :param classes: list of class sequences (list of lists)
    :param num_classes: total number of available classes
    :return: torch.tensor of shape (batch_size, max_seq_len, num_classes)
    :raise ValueError: if there is any class value that is negative or >= num_classes
    """

    eye = torch.eye(num_classes, dtype=torch.float32)
    try:
        tensors = [eye[class_seq] for class_seq in classes]
    except IndexError:
        msg = f'Every class must be a non-negative numbers less than num_classes={num_classes}. ' \
              f'Got classes {classes}'
        raise ValueError(msg)
    return torch.stack(tensors)

# todo: allow any kind of punctuation


"""
    (Remove this later or use it)
    Add special extra tokens <Start> and <End> to the
    beginning and the end of list respectively.
"""