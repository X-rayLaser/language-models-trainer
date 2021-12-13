from functools import reduce
from operator import add


def tokenize_prompt(prompt, allowed_punctuation='!?.,:;-+#$%'):
    """Turns a raw prompt text into a sequence of tokens.

    Usage:
    >>>tokenize_prompt('Hello world!')
    >>>['Hello', 'world', '!']

    :param prompt: str
    :param allowed_punctuation: an iterable of handled punctuation characters
    :return: list[str]
    """
    texts = prompt.split(' ')
    tokens = [[s] if s.isalpha() else [s[:-1], s[-1]] for s in texts if s]
    initial = []
    tokens = reduce(add, tokens, initial)
    return [t for t in tokens if t.isalpha() or t in allowed_punctuation]


def paragraph_to_tokens(paragraph, allowed='!?.,:;-+#$%'):
    return [token for sentence in paragraph for token in sentence
            if token.isalpha() or token in allowed]


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

    def decode(self, number):
        return self.vocab[number] if 0 <= number < len(self.vocab) else self.oov_token

    def serialize(self):
        return ' '.join(self.vocab)

    def _validate_vocab(self, vocab):
        with_spaces = [word for word in vocab if ' ' in word]
        if any(with_spaces):
            raise ValueError(f'Vocab words must not contain whitespaces. '
                             f'Whitespace words: {with_spaces}')


# todo: allow any kind of punctuation


"""
    (Remove this later or use it)
    Add special extra tokens <Start> and <End> to the
    beginning and the end of list respectively.
"""