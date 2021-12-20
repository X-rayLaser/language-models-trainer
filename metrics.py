from torch.nn import functional as F
from functools import reduce
from operator import mul


def batch_perplexity(y_hat, ground_true, lengths):
    triples = zip(y_hat, ground_true, lengths)
    pps = [perplexity(y[:size], classes[:size]) for y, classes, size in triples]
    return sum(pps) / len(pps)


def nth_root(x, n):
    return x ** (1. / n)


def perplexity(y_hat, ground_true):
    """Perplexity per single example

    :param y_hat: tensor of probabilities of shape (num_steps, num_classes)
    :param ground_true: tensor of ground true labels of shape (num_steps,)
    :return: a perplexity as a scalar
    """
    row_indices = list(range(len(ground_true)))
    col_indices = ground_true
    probabilities = y_hat[row_indices, col_indices]
    likelihood = reduce(mul, probabilities, 1).item()

    n = len(ground_true)

    eps = 10**(-10)
    return 1. / (nth_root(likelihood, n) + eps)
