import torch


def batch_perplexity(y_hat, ground_true, lengths):
    triples = zip(y_hat, ground_true, lengths)
    pps = [perplexity(y[:size], classes[:size]) for y, classes, size in triples]
    return sum(pps) / len(pps)


def perplexity(y_hat, ground_true):
    """Perplexity per single example

    :param y_hat: tensor of probabilities of shape (num_steps, num_classes)
    :param ground_true: tensor of ground true labels of shape (num_steps,)
    :return: a perplexity as a scalar
    """
    row_indices = list(range(len(ground_true)))
    col_indices = ground_true
    probabilities = y_hat[row_indices, col_indices]
    n = len(ground_true)
    return torch.exp(-torch.log(probabilities).sum() / n).item()
