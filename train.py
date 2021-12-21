# use naive split strategy to split the whole corpus into smaller pieces (of 10 sentences)
# or run truncated backpropagation through time:
# update weights every k1 steps by running BPTT for k2 steps back
# for instance, in default untruncated version, k1 = 1 and k2 goes all the way to step=t1
# (weights are updated on every step from t1 to T, and to compute the update for the step t,
# all previous step calculations are considered
# some users suggest to use detach on outputs h of LSTM layer
import os
import argparse

import nltk
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from preprocessing import Encoder, ParagraphsDataset, build_vocab, Collator, flatten_paragraphs, wrapped_paragraphs
from utils import run_training_loop, ModelStorage
from model import Net


def get_paragraphs_for(category=None):
    nltk.download('brown')
    return nltk.corpus.brown.paras(categories=category)


def train(*,  # function accepts keyword only arguments
          size=None, lstm_cells=32,
          train_fraction=0.9, batch_size=8,
          genre='fiction', vocab_size=20000, save_dir='checkpoints',
          epochs=100):
    all_paragraphs = get_paragraphs_for(genre)
    if size:
        all_paragraphs = all_paragraphs[:size]

    train_size = int(len(all_paragraphs) * train_fraction)
    train_paragraphs = all_paragraphs[:train_size]
    test_paragraphs = all_paragraphs[train_size:]
    print(f'Training examples {train_size}')

    vocab = build_vocab(flatten_paragraphs(train_paragraphs), max_size=vocab_size)

    encoder = Encoder.build(vocab)
    print('Unique tokens total:', len(encoder))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device', device)

    collate = Collator(num_classes=len(encoder), device=device)
    train_dataset = ParagraphsDataset(wrapped_paragraphs(train_paragraphs), encoder)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    val_dataset = ParagraphsDataset(wrapped_paragraphs(test_paragraphs), encoder)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    num_training_batches = len(train_dataloader)
    iterations_per_metric = max(1, num_training_batches // 10)

    print(f'# of training batches {num_training_batches}, '
          f'# of validation batches {len(val_dataloader)}')

    net_params = dict(num_classes=len(encoder), hidden_dim=lstm_cells)
    net = Net(**net_params)
    net.to(device)

    optimizer = optim.RMSprop(net.parameters(), lr=0.001)

    def save_callback(epoch):
        path = os.path.join(save_dir, f'epoch_{epoch}')
        ModelStorage.save(net, net_params, encoder, path)

    run_training_loop(net=net, optimizer=optimizer, train_loader=train_dataloader,
                      val_loader=val_dataloader, on_epoch=save_callback,
                      epochs=epochs, iterations_per_metric=iterations_per_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a language model on a given literature genre')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Path to a location where a model will be saved to')
    parser.add_argument('--capacity', type=int, default=32, help='Model capacity (size of the LSTM layer)')

    parser.add_argument('--genre', type=str, default='fiction',
                        help='Literature genre used to train a language model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Maximum number of words in a vocabulary (defines a total # of unique input encodings)')
    parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
    parser.add_argument('--prompt', type=str, default='', help='Text used to condition a model on')
    args = parser.parse_args()

    train(lstm_cells=args.capacity, batch_size=args.batch_size, genre=args.genre,
          vocab_size=args.vocab_size, save_dir=args.save_dir, epochs=args.epochs)
