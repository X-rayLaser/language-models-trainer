import os
import nltk
from preprocessing import Paragraph, Sentence


class NltkDataset:
    def __init__(self, corpus_name, size=None, categories=None, train_fraction=0.9, as_paragraphs=True):
        corpus = getattr(nltk.corpus, corpus_name)
        val_fraction = (1 - train_fraction) / 2.
        if as_paragraphs:
            all_fragments = [Paragraph(para) for para in corpus.paras(categories=categories)]
        else:
            all_fragments = [Sentence(sent) for sent in corpus.sents(categories=categories)]

        if size:
            all_fragments = all_fragments[:size]

        train_size = int(len(all_fragments) * train_fraction)
        val_size = int(len(all_fragments) * val_fraction)

        self.train_fragments = all_fragments[:train_size]
        self.val_fragments = all_fragments[train_size: train_size + val_size]
        self.test_fragments = all_fragments[train_size + val_size:]

    def get_training_fragments(self):
        return self.train_fragments

    def get_validation_fragments(self):
        return self.val_fragments

    def get_test_fragments(self):
        return self.test_fragments


class Wiki2:
    def __init__(self, ds_dir, size=None, as_paragraphs=True):
        self.get_fragments = get_paragraphs if as_paragraphs else get_sentences
        if not os.path.isdir(ds_dir):
            raise NotADirectoryError(f'{ds_dir} is not a directory')

        self.size = size
        self.training_path = os.path.join(ds_dir, 'wiki.train.tokens')
        self.validation_path = os.path.join(ds_dir, 'wiki.valid.tokens')
        self.test_path = os.path.join(ds_dir, 'wiki.test.tokens')

    def get_training_fragments(self):
        fragments = self.get_fragments(self.training_path)
        if self.size:
            fragments = fragments[:self.size]
        return fragments

    def get_validation_fragments(self):
        return self.get_fragments(self.validation_path)

    def get_test_fragments(self):
        return self.get_fragments(self.test_path)


def get_paragraphs(path):
    with open(path, 'r') as f:
        text = f.read()

    paragraphs = [para.strip().split(' ') for para in text.split('\n') if para.strip() and para.strip()[0] != '=']

    paragraphs_of_sentences = []
    for tokens in paragraphs:
        start = 0
        sentences = []
        for i, token in enumerate(tokens):
            if token in '.?!':
                sentences.append(tokens[start:i + 1])
                start = i + 1
        paragraphs_of_sentences.append(Paragraph(sentences))

    return paragraphs_of_sentences


def get_sentences(path):
    return list(Sentence(sent) for para in get_paragraphs(path) for sent in para.sentences)
