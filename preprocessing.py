from functools import reduce
from operator import add
from collections import Counter

import torch
from torch.utils.data import Dataset


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


wrapped_tokens.start_token = '<start>'
wrapped_tokens.end_token = '<end>'


def lower_case_tokens(tokens):
    return [t.lower() for t in tokens]


def clean(tokens, allowed_punctuation='!?.,:;-+#$%'):
    return [t for t in tokens if t.isalpha() or t in allowed_punctuation]


def prepare_paragraph(para):
    tokens = paragraph_to_tokens(para)
    cleaned_tokens = clean(tokens)
    return lower_case_tokens(wrapped_tokens(cleaned_tokens))


def wrapped_paragraphs(paragraphs):
    return [prepare_paragraph(para) for para in paragraphs]


def flatten_paragraphs(paragraphs):
    for para in paragraphs:
        for token in prepare_paragraph(para):
            yield token


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


class ParagraphsDataset(Dataset):
    def __init__(self, paragraphs, encoder):
        self.paragraphs = paragraphs
        self.encoder = encoder

    def __getitem__(self, index):
        paragraph = self.paragraphs[index]
        return ExampleFactory(paragraph, self.encoder).make_example()

    def __len__(self):
        return len(self.paragraphs)


class Mask:
    def __init__(self, lengths, max_length):
        self.mask = torch.zeros(len(lengths), max_length, dtype=torch.bool)
        self.lengths = lengths

        for i, length in enumerate(lengths):
            self.mask[i, :length] = True


def add_padding(seq, size, filler):
    seq = list(seq)
    while len(seq) < size:
        seq.append(filler)
    return seq


def pad_sequences(seqs, filler):
    lengths = [len(seq) for seq in seqs]
    max_length = max(lengths)

    mask = Mask(lengths, max_length)

    padded = [add_padding(seq, max_length, filler) for seq in seqs]
    return padded, mask


class Collator:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device

    def __call__(self, batch):
        inputs = [x for x, y in batch]
        targets = [y for x, y in batch]

        filler = targets[0][-1]
        inputs, _ = pad_sequences(inputs, filler)
        targets, mask = pad_sequences(targets, filler)
        mask.mask = mask.mask.to(self.device)

        num_classes = self.num_classes

        return (one_hot_tensor(inputs, num_classes).to(self.device),
                torch.tensor(targets, dtype=torch.long).to(self.device),
                mask)


def one_hot_tensor(classes, num_classes):
    """Form a 1-hot tensor from a list of class sequences

    :param classes: list of class sequences (list of lists)
    :param num_classes: total number of available classes
    :return: torch.tensor of shape (batch_size, max_seq_len, num_classes)
    :raise ValueError: if there is any class value that is negative or >= num_classes
    """

    # todo: clean this up (make this function into a callable class)

    if not hasattr(one_hot_tensor, 'eye'):
        one_hot_tensor.eye = {}

    if num_classes not in one_hot_tensor.eye:
        one_hot_tensor.eye[num_classes] = torch.eye(num_classes, dtype=torch.float32)
    eye = one_hot_tensor.eye[num_classes]
    try:
        tensors = [eye[class_seq] for class_seq in classes]
    except IndexError:
        msg = f'Every class must be a non-negative number less than num_classes={num_classes}. ' \
              f'Got classes {classes}'
        raise ValueError(msg)

    return torch.stack(tensors)

# todo: allow any kind of punctuation
