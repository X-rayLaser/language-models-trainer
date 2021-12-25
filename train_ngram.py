import math
import os
from collections import Counter, defaultdict
import shelve

from nltk import ngrams
import numpy as np

from datasets import NltkDataset
from preprocessing import Paragraph, Encoder, build_vocab, tokenize_prompt, Sentence


class NGramModel:
    def __init__(self, counts_ensemble, smoothing=True):
        self.counts_ensemble = counts_ensemble
        self.n = self.counts_ensemble.n
        self.smoothing = smoothing

    def probabilities(self, tokens):
        return (self.p_next(prefix) for *prefix, _ in ngrams(tokens, self.n))

    def log_probability(self, tokens):
        eps = 10**(-10)
        return sum(math.log(self.p_next(prefix)[token] + eps)
                   for *prefix, token in ngrams(tokens, self.n))

    def probability(self, tokens):
        return math.exp(self.log_probability(tokens))

    def _kneser_ney(self, prefix, token):
        pass

    def _stupid_backoff_prob(self, prefix, token):
        # todo: fix this
        n = self.n
        prev_tokens = tuple(prefix)
        multiplier = 1
        lam = 0.01
        while n > 0:
            count_table = self.count_tables[n - 1]
            prefix = prev_tokens[-n:]
            try:
                pmf = count_table[prefix]
                if pmf[token] > 0:
                    return multiplier * pmf[token]
                n -= 1
                multiplier *= lam
            except KeyError:
                n -= 1
                multiplier *= lam

        unigram_table = self.count_tables[0]

        assert len(unigram_table) == 1

        return multiplier * next(iter(unigram_table.values()))[token]

    def p_next(self, prev_tokens):
        # todo: maybe add smoothing
        counts_row = self.counts_ensemble.get_counts_row(prev_tokens)

        if self.smoothing:
            vocab_size = counts_row.shape[0]
            k = 0.01
            p = (counts_row + k) / (k * vocab_size + counts_row.sum())
            return p
        return counts_row / counts_row.sum()

    def generate(self, prompt, size):
        prefix_size = self.n - 1
        tail = prompt[-prefix_size:] if prefix_size > 0 else []
        outputs = []
        for _ in range(size):
            y = self.sample_next(tail)
            outputs.append(y)
            tail = tail[1:] + [y] if prefix_size > 0 else []

        return outputs, prompt + outputs

    def sample_next(self, prev_tokens):
        pmf = self.p_next(prev_tokens)
        num_classes = pmf.shape[0]
        return np.random.choice(num_classes, p=pmf)


def back_off(counts):
    # todo: fix this and use this as a main routine to compute n-1 grams to speed up build_ensemble_counts function
    n_1_table = {}
    for key in counts:
        tail = key[1:]
        if tail not in n_1_table:
            n_1_table[tail] = [0] * 10000
        n_1_table[tail] += counts[key]


def padded_ngram_tokens(fragments, n):
    padding_size = n - 1
    for fragment in fragments:
        tokens = fragment.prepare_tokens()
        start_token, *remainder = tokens
        yield from [start_token] * padding_size
        yield from remainder


def build_counts_table(classes, num_classes, n, save_path):
    if n == 0:
        raise ValueError(f'Wrong ngram order: {n}')
    #counts = Counter(ngrams(classes, n))

    with shelve.open(save_path) as count_table:
        for *n_1_gram, count_class in ngrams(classes, n):
            key = '_'.join(map(str, n_1_gram))
            if key not in count_table:
                count_table[key] = np.zeros(num_classes, dtype=np.int32)

            counts_row = count_table[key]
            counts_row[count_class] += 1
            count_table[key] = counts_row
    print('Done', n)


class CountTableEnsemble:
    @classmethod
    def build_counts(cls, train_fragments, encoder, n, save_dir):
        num_classes = len(encoder)

        def classes_gen():
            return encoded_classes(train_fragments, encoder, n)

        os.makedirs(save_dir, exist_ok=True)
        for i in range(n):
            file_name = f'table_{i + 1}'
            save_path = os.path.join(save_dir, file_name)
            build_counts_table(classes_gen(), num_classes,
                               n=i + 1, save_path=save_path)

    def __init__(self, dir_path):
        shelve_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        shelve_paths.sort(key=lambda name: int(name.split('_')[-1]))
        self.count_tables = [shelve.open(f) for f in shelve_paths if os.path.isfile(f)]
        self.n = len(self.count_tables)

    def get_counts_row(self, prev_tokens):
        n = self.n

        while n > 0:
            count_table = self.count_tables[n - 1]
            prefix = '_'.join(map(str, prev_tokens[-n:]))

            try:
                return count_table[prefix]
            except KeyError:
                n -= 1

        unigram_table = self.count_tables[0]

        assert len(unigram_table) == 1

        return next(iter(unigram_table.values()))

    def close(self):
        for t in self.count_tables:
            t.close()


def encoded_classes(fragments, encoder, n):
    return (encoder.encode(token) for token in padded_ngram_tokens(fragments, n))


ds = NltkDataset('brown', categories=None)
train_fragments = ds.get_training_fragments()
val_fragments = ds.get_validation_fragments()
test_fragments = ds.get_test_fragments()

ngram_order = 4

vocab = build_vocab(padded_ngram_tokens(train_fragments, ngram_order), max_size=10000)
encoder = Encoder.build(vocab)
print('vocab size', len(vocab))
save_dir = 'counts_ensemble'
CountTableEnsemble.build_counts(train_fragments, encoder, n=ngram_order, save_dir=save_dir)
count_tables = CountTableEnsemble(save_dir)

model = NGramModel(count_tables, smoothing=False)

training = encoded_classes(train_fragments, encoder, ngram_order)
validation = encoded_classes(val_fragments, encoder, ngram_order)
test = encoded_classes(test_fragments, encoder, ngram_order)


def calculate_perplexity(model, tokens):
    n = len(tokens) - ngram_order
    logp = model.log_probability(tokens)
    return np.exp(- logp / n)


train_pp = calculate_perplexity(model, list(training)[:1000])
val_pp = calculate_perplexity(model, list(validation)[:1000])
print(train_pp)
print(val_pp)


while True:
    prompt = input('Enter text: ')
    tokens = tokenize_prompt(prompt)
    sentence = Sentence(tokens)

    clean_tokens = list(padded_ngram_tokens([sentence], ngram_order))[:-1]

    clean_tokens = encoder.encode_many(clean_tokens)
    outputs, _ = model.generate(clean_tokens, size=50)
    res = encoder.decode_many(outputs)

    print('Prompt:', prompt)
    print('Generated:', res)


# todo: separate training from evaluation and sampling
# todo: faster algorithm for building count tables (do not scan the whole dataset multiple times, batch write)
# todo: implement effective smoothing techniques
