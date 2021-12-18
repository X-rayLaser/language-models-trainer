# use naive split strategy to split the whole corpus into smaller pieces (of 10 sentences)
# or run truncated backpropagation through time:
# update weights every k1 steps by running BPTT for k2 steps back
# for instance, in default untruncated version, k1 = 1 and k2 goes all the way to step=t1
# (weights are updated on every step from t1 to T, and to compute the update for the step t,
# all previous step calculations are considered
# some users suggest to use detach on outputs h of LSTM layer
import nltk
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from preprocessing import paragraph_to_tokens, Encoder, wrapped_tokens, ParagraphsDataset, build_vocab, Collator
from model import Net


def get_paragraphs_for(category=None):
    return nltk.corpus.brown.paras(categories=category)


train_fraction = 0.9
batch_size = 20
genre = 'fiction'
max_vocab_size = 20000

all_paragraphs = get_paragraphs_for(genre)
train_size = int(len(all_paragraphs) * train_fraction)
train_paragraphs = all_paragraphs[:train_size]
test_paragraphs = all_paragraphs[train_size:]
print(f'Training examples {train_size}')


def lower_case_tokens(tokens):
    return [t.lower() for t in tokens]


def prepare_paragraph(para):
    return lower_case_tokens(wrapped_tokens(paragraph_to_tokens(para)))


def wrapped_paragraphs(paragraphs):
    return [prepare_paragraph(para) for para in paragraphs]


def flatten_paragraphs(paragraphs):
    for para in paragraphs:
        for token in prepare_paragraph(para):
            yield token


class MaskedCrossEntropy:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, y_hat, ground_true, mask):
        losses = self.loss_function(self.swap_axes(y_hat), self.swap_axes(ground_true))
        return losses[mask].mean()

    def swap_axes(self, t):
        return t.transpose(1, 2)


vocab = build_vocab(flatten_paragraphs(train_paragraphs), max_size=max_vocab_size)

encoder = Encoder.build(vocab)

collate = Collator(len(encoder))
train_dataset = ParagraphsDataset(wrapped_paragraphs(train_paragraphs), encoder)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)

test_dataset = ParagraphsDataset(wrapped_paragraphs(test_paragraphs), encoder)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

net = Net(num_classes=len(encoder), hidden_dim=128)

criterion = MaskedCrossEntropy()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for i, batch in enumerate(train_dataloader):
    inputs, labels, mask = batch
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)

    loss = criterion(outputs, labels, mask)

    loss.backward()
    optimizer.step()

    if i % 1 == 0:
        print(f'{i} done')


# todo: refactor, separate data preparation code for ease of testing (4 layers of representation: paragraphs, tokens, numbers, vectors)
# todo: cleaning pipeline (extract tokens from raw user entered text, remove forbidden chars, prepend with <start> token)
# todo: extra cleaning: to lower case
# todo: apply pipeline to nltk brown dataset (when getting tokens from paragraph)
# todo: put everything together, create LSTM model, training loop and metrics (accuracy, perplexity)
