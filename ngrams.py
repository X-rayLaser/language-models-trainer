import math
import os
from collections import defaultdict, UserDict
import shelve

from nltk import ngrams
import numpy as np

from preprocessing import Encoder, build_vocab, tokenize_prompt, Sentence


class SerializableModel:
    def save_model(self, path):
        raise NotImplementedError

    @classmethod
    def load_saved_model(cls, path, model_params):
        raise NotImplementedError


class NGramModel(SerializableModel):
    counts_dir = 'counts_ensemble'

    @classmethod
    def build_model(cls, dataset, save_dir, ngram_order=4, max_size=5000):
        train_fragments = dataset.get_training_fragments()

        vocab = build_vocab(padded_ngram_tokens(train_fragments, ngram_order), max_size=max_size)
        encoder = Encoder.build(vocab)
        print('vocab size', len(vocab))

        counts_dir_path = os.path.join(save_dir, cls.counts_dir)
        CountTableEnsemble.build_counts(train_fragments, encoder, ngram_order, counts_dir_path)
        return NGramModel(save_dir, smoothing=False), encoder

    @classmethod
    def load_saved_model(cls, path, model_params):
        return cls(**model_params)

    def __init__(self, save_dir, smoothing=True):
        counts_dir_path = os.path.join(save_dir, self.counts_dir)
        self.counts_ensemble = CountTableEnsemble(counts_dir_path)
        self.n = self.counts_ensemble.n
        self.smoothing = smoothing

    def close(self):
        self.counts_ensemble.close()

    def save_model(self, path):
        # nothing to do because count tables were already created after building the model
        pass

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
            count_table = self.counts_ensemble[n - 1]
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

        unigram_table = self.counts_ensemble[0]

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


def reduce_order(counts, output_table):
    # todo: fix this and use this as a main routine to compute n-1 grams to speed up build_ensemble_counts function
    n_1_table = output_table
    counts_row = next(iter(counts.values()))
    num_classes = counts_row.shape[0]

    max_cache_size = 1000
    in_memory_table = defaultdict(lambda: np.zeros(num_classes, dtype=np.int32))

    for key in counts:
        n_1_gram = key.split('_')

        tail = '_'.join(n_1_gram[1:])
        if len(in_memory_table) < max_cache_size or tail in in_memory_table:
            in_memory_table[tail] += counts[key]
        else:
            if tail not in n_1_table:
                n_1_table[tail] = np.zeros(num_classes, dtype=np.int32)
            n_1_table[tail] += counts[key]


def padded_ngram_tokens(fragments, n):
    padding_size = n - 1
    for fragment in fragments:
        tokens = fragment.prepare_tokens()
        start_token, *remainder = tokens
        yield from [start_token] * padding_size
        yield from remainder


def build_counts_table3(classes, num_classes, n, save_path):
    max_cache_size = 1000000

    in_memory_table = defaultdict(lambda: SparseArray(num_classes))

    count_table = shelve.open(save_path)
    for i, (*n_1_gram, count_class) in enumerate(ngrams(classes, n)):
        key = '_'.join(map(str, n_1_gram))

        if i % 1000 == 0:
            print(f'\rBuilding counts table, done {i} iterations', end='')

        if len(in_memory_table) < max_cache_size or key in in_memory_table:
            counts_row = in_memory_table[key]
            counts_row.increment_count(count_class)
        else:
            if key not in count_table:
                count_table[key] = np.zeros(num_classes, dtype=np.int32)

            counts_row = count_table[key]
            counts_row[count_class] += 1
            count_table[key] = counts_row

    count_table.update(in_memory_table)

    count_table.close()

    print('\rDone')


class SparseArray(UserDict):
    def __init__(self, dense_size):
        super().__init__()
        self.indices = {}
        self.counts = []
        self.dense_size = dense_size

    def __setitem__(self, key, value):
        if key not in self.indices:
            cur_size = len(self.indices)
            self.indices[key] = cur_size
            self.counts.append(0)
        idx = self.indices[key]
        self.counts[idx] = value

    def __getitem__(self, key):
        idx = self.indices[key]
        return self.counts[idx]

    def increment_count(self, token):
        self[token] = self.get(token, 0) + 1

    def to_numpy(self):
        dense = np.zeros(self.dense_size, dtype=np.int32)
        for token, idx in self.indices.items():
            dense[token] = self.counts[idx]

        return dense

    def tolist(self):
        return self.to_numpy().tolist()


class CountTable(UserDict):
    def __init__(self, save_path):
        super().__init__()
        self.d = shelve.open(save_path)

    def __getitem__(self, item):
        key = '_'.join(map(str, item))
        return self.d[key]

    def __len__(self):
        return len(self.d)

    def close(self):
        self.d.close()

    def any_value(self):
        return next(iter(self.d.values()))


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
            build_counts_table3(classes_gen(), num_classes,
                               n=i + 1, save_path=save_path)

    def _experimental(self):
        file_name = f'table_{n}'
        save_path = os.path.join(save_dir, file_name)
        counts_table = build_counts_table(classes_gen(), num_classes, n=n, save_path=save_path)

        for i in range(n - 1, 0, -1):
            file_name = f'table_{i}'
            save_path = os.path.join(save_dir, file_name)
            lower_order_table = shelve.open(save_path)
            reduce_order(counts_table, lower_order_table)
            counts_table.close()
            counts_table = lower_order_table

        counts_table.close()
        return

    def __init__(self, dir_path):
        shelve_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

        def get_file_name(path):
            _, file_name = os.path.split(path)
            print(file_name)
            return file_name

        shelve_paths.sort(key=lambda path: int(get_file_name(path).split('_')[-1]))
        self.count_tables = [CountTable(f) for f in shelve_paths if os.path.isfile(f)]
        self.n = len(self.count_tables)

    def get_counts_row(self, prev_tokens):
        n = self.n

        while n > 0:
            count_table = self.count_tables[n - 1]
            prefix = prev_tokens[-n:]

            try:
                return count_table[prefix].to_numpy()
            except KeyError:
                n -= 1

        unigram_table = self.count_tables[0]

        assert len(unigram_table) == 1

        return unigram_table.any_value().to_numpy()

    def close(self):
        for t in self.count_tables:
            t.close()


def encoded_classes(fragments, encoder, n):
    return (encoder.encode(token) for token in padded_ngram_tokens(fragments, n))


def calculate_perplexity(model, tokens, ngram_order):
    n = len(tokens) - ngram_order
    logp = model.log_probability(tokens)
    return np.exp(- logp / n)


def sample(model, encoder, prompt, ngram_order):
    tokens = tokenize_prompt(prompt)
    sentence = Sentence(tokens)

    clean_tokens = list(padded_ngram_tokens([sentence], ngram_order))[:-1]

    clean_tokens = encoder.encode_many(clean_tokens)
    outputs, _ = model.generate(clean_tokens, size=50)
    return encoder.decode_many(outputs)


# todo: having to manually call .close is error-prone
# todo: one class for creating counts table, the other for reading it, tests
