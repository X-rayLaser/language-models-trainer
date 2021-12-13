# use naive split strategy to split the whole corpus into smaller pieces (of 10 sentences)
# or run truncated backpropagation through time:
# update weights every k1 steps by running BPTT for k2 steps back
# for instance, in default untruncated version, k1 = 1 and k2 goes all the way to step=t1
# (weights are updated on every step from t1 to T, and to compute the update for the step t,
# all previous step calculations are considered
# some users suggest to use detach on outputs h of LSTM layer
import nltk
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from preprocessing import paragraph_to_tokens, Encoder, wrapped_tokens, ExampleFactory, build_vocab, Collator


def get_paragraphs_for(category=None):
    return nltk.corpus.brown.paras(categories=category)


class ParagraphsDataset(Dataset):
    def __init__(self, paragraphs, encoder):
        self.paras = [ExampleFactory(para, encoder) for para in paragraphs]

    def __getitem__(self, index):
        paragraph = self.paras[index]
        return paragraph.make_example()

    def __len__(self):
        return len(self.paras)


train_fraction = 0.9
batch_size = 20
genre = 'fiction'
max_vocab_size = 20000

all_paragraphs = get_paragraphs_for(genre)
train_size = len(all_paragraphs) * train_fraction
train_paragraphs = all_paragraphs[:train_size]
test_paragraphs = all_paragraphs[train_size:]


def wrapped_paragraphs(paragraphs):
    return (wrapped_tokens(paragraph_to_tokens(para)) for para in paragraphs)


vocab = build_vocab(wrapped_paragraphs(train_paragraphs), max_size=max_vocab_size)

encoder = Encoder.build(vocab)

collate = Collator(len(encoder))
train_dataset = ParagraphsDataset(wrapped_paragraphs(train_paragraphs), encoder)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)

test_dataset = ParagraphsDataset(wrapped_paragraphs(test_paragraphs), encoder)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)


for inputs, outputs in train_dataloader:
    pass


# todo: refactor, separate data preparation code for ease of testing (4 layers of representation: paragraphs, tokens, numbers, vectors)
# todo: cleaning pipeline (extract tokens from raw user entered text, remove forbidden chars, prepend with <start> token)
# todo: extra cleaning: to lower case
# todo: apply pipeline to nltk brown dataset (when getting tokens from paragraph)
# todo: test dataset class
# todo: put everything together, create LSTM model, training loop and metrics (accuracy, perplexity)
