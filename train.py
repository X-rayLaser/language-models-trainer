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

from preprocessing import Encoder, FragmentsDataset, build_vocab, Collator, Paragraph, Sentence
from utils import run_training_loop, ModelStorage
from model import Net
from datasets import NltkDataset, Wiki2


def get_paragraphs_for(category=None):
    nltk.download('brown')
    return [Paragraph(para) for para in nltk.corpus.brown.paras(categories=category)]


def get_sentences_for(category=None):
    nltk.download('brown')
    return [Sentence(sent) for sent in nltk.corpus.brown.sents(categories=category)]


def train(*,  # function accepts keyword only arguments
          dataset,
          lstm_cells=32,
          batch_size=8,
          vocab_size=20000, save_dir='checkpoints',
          epochs=100):

    print(f'Fetching examples using {dataset.__class__.__name__}')

    train_fragments = dataset.get_training_fragments()
    test_fragments = dataset.get_validation_fragments() + dataset.get_test_fragments()

    print(f'Training examples {len(train_fragments)}')

    vocab = build_vocab(Paragraph.flatten_fragments(train_fragments), max_size=vocab_size)

    encoder = Encoder.build(vocab)
    print('Unique tokens total:', len(encoder))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device', device)

    collate = Collator(num_classes=len(encoder), device=device)
    train_dataset = FragmentsDataset(Paragraph.prepare_fragments(train_fragments), encoder)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    val_dataset = FragmentsDataset(Paragraph.prepare_fragments(test_fragments), encoder)
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

    parser.add_argument('--paras', type=bool, default=False,
                        help='When set, paragraphs will be used for training. Sentences are used by default')
    parser.add_argument('--size', type=int, default=None, help='Total number of examples in train/validation split')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Path to a location where a model will be saved to')
    parser.add_argument('--capacity', type=int, default=32, help='Model capacity (size of the LSTM layer)')

    parser.add_argument('--genre', type=str, default='',
                        help='Literature genre used to train a language model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Maximum number of words in a vocabulary (defines a total # of unique input encodings)')
    parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
    parser.add_argument('--prompt', type=str, default='', help='Text used to condition a model on')
    args = parser.parse_args()

    genre = args.genre or None
    if genre == 'wiki2':
        ds = Wiki2('corpora/wikitext-2', size=args.size, as_paragraphs=args.paras)
    else:
        ds = NltkDataset(corpus_name='brown', size=args.size,
                         categories=genre, as_paragraphs=args.paras)

    train(dataset=ds,
          lstm_cells=args.capacity,
          batch_size=args.batch_size,
          vocab_size=args.vocab_size, save_dir=args.save_dir, epochs=args.epochs)


# todo: add Penn Tree Bank dataset class
# todo: script for evaluating perplexity on a corpus
# todo: fine-tune GPT-2 or T5
# todo: different implementation of dataset (the one that can handle huge corpora)
# todo: get more data
