import argparse

from datasets import NltkDataset
from ngrams import calculate_perplexity, encoded_classes
from utils import NgramStorage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a N-gram model with respect to a given metric')
    parser.add_argument('ngram_path', type=str, help='Path to a pre-trained N-gram model')
    parser.add_argument('--n', default=1, type=int, help='Order of N-gram model (1 for unigrams, 2 for bigrams, etc')
    args = parser.parse_args()

    model, encoder = NgramStorage.load(args.ngram_path)

    ds = NltkDataset('brown', categories='fiction')
    train_fragments = ds.get_training_fragments()
    val_fragments = ds.get_validation_fragments()
    test_fragments = ds.get_test_fragments()

    ngram_order = args.n
    training = encoded_classes(train_fragments, encoder, ngram_order)
    validation = encoded_classes(val_fragments, encoder, ngram_order)
    test = encoded_classes(test_fragments, encoder, ngram_order)

    train_pp = calculate_perplexity(model, list(training)[:1000], ngram_order)
    val_pp = calculate_perplexity(model, list(validation)[:1000], ngram_order)
    print('Training data perplexity', train_pp)
    print('Validation data perplexity', val_pp)
