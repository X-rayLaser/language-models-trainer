# use naive split strategy to split the whole corpus into smaller pieces (of 10 sentences)
# or run truncated backpropagation through time:
# update weights every k1 steps by running BPTT for k2 steps back
# for instance, in default untruncated version, k1 = 1 and k2 goes all the way to step=t1
# (weights are updated on every step from t1 to T, and to compute the update for the step t,
# all previous step calculations are considered
# some users suggest to use detach on outputs h of LSTM layer
import nltk
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from preprocessing import Encoder, ParagraphsDataset, build_vocab, Collator, flatten_paragraphs, wrapped_paragraphs
from utils import MaskedCrossEntropy, run_training_loop
from model import Net


def get_paragraphs_for(category=None):
    return nltk.corpus.brown.paras(categories=category)


train_fraction = 0.9
batch_size = 20
genre = 'fiction'
max_vocab_size = 20000

all_paragraphs = get_paragraphs_for(genre)[:40]
train_size = int(len(all_paragraphs) * train_fraction)
train_paragraphs = all_paragraphs[:train_size]
test_paragraphs = all_paragraphs[train_size:]
print(f'Training examples {train_size}')

vocab = build_vocab(flatten_paragraphs(train_paragraphs), max_size=max_vocab_size)

encoder = Encoder.build(vocab)
print(len(encoder))
collate = Collator(len(encoder))
train_dataset = ParagraphsDataset(wrapped_paragraphs(train_paragraphs), encoder)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate)

test_dataset = ParagraphsDataset(wrapped_paragraphs(test_paragraphs), encoder)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

net = Net(num_classes=len(encoder), hidden_dim=128)

criterion = MaskedCrossEntropy()
optimizer = optim.RMSprop(net.parameters(), lr=0.001)

run_training_loop(net, optimizer, criterion, train_dataloader, encoder, epochs=100)


from utils import sample
for i in range(100):
    prompt = input('Enter a text')
    print(sample(net, encoder, prompt, steps=10))

# todo: refactor
# todo: clean nltk paragraphs
# todo: training loop and metrics (accuracy, perplexity)
# todo: condition on the first word
