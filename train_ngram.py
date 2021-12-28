import argparse

from datasets import NltkDataset
from ngrams import NGramModel
from utils import NgramStorage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a N-gram model with respect to a given metric')
    parser.add_argument('save_dir', type=str, help='Path to a pre-trained N-gram model')
    parser.add_argument('--n', default=1, type=int, help='Order of N-gram model (1 for unigrams, 2 for bigrams, etc')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Maximum number of words in a vocabulary (defines a total # of unique input encodings)')
    args = parser.parse_args()

    ds = NltkDataset('brown', categories=None)
    train_fragments = ds.get_training_fragments()
    val_fragments = ds.get_validation_fragments()
    test_fragments = ds.get_test_fragments()

    save_dir = args.save_dir

    model, encoder = NGramModel.build_model(ds, save_dir, ngram_order=args.n, max_size=args.vocab_size)
    params = dict(save_dir=save_dir, smoothing=False)
    NgramStorage.save(model, params, encoder, save_dir)
    model.close()


# todo: faster algorithm for building count tables (do not scan the whole dataset multiple times, batch write)
# todo: implement effective smoothing techniques
