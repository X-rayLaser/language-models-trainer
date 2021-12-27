import time
import os
import ngrams
from ngrams import padded_ngram_tokens
from preprocessing import build_vocab, Encoder
from datasets import NltkDataset


ds = NltkDataset('brown', categories=None)
train_fragments = ds.get_training_fragments()[:2000]
ngram_order = 4
max_size = 10000

save_path = 'test_count_table'
vocab = build_vocab(padded_ngram_tokens(train_fragments, ngram_order), max_size=max_size)
print('vocab size', len(vocab))
encoder = Encoder.build(vocab)
classes = ngrams.encoded_classes(train_fragments, encoder, ngram_order)
num_classes = len(encoder)


if os.path.isfile(save_path):
    os.remove(save_path)


t = time.time()
ngrams.build_counts_table3(classes, num_classes, ngram_order, save_path)
print('elapsed seconds:', time.time() - t)
