import os
import json
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from metrics import batch_perplexity
from preprocessing import tokenize_prompt, wrapped_tokens, one_hot_tensor, Encoder
from model import Net


class MovingAverage:
    def __init__(self, nun_points):
        self.num_points = nun_points
        self._history = deque()

    def __call__(self, x):
        self._history.append(x)
        if len(self._history) > self.num_points:
            self._history.popleft()
        return sum(self._history) / len(self._history)


class MaskedCrossEntropy:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, y_hat, ground_true, mask):
        losses = self.loss_function(self.swap_axes(y_hat), ground_true)
        return losses[mask].mean()

    def swap_axes(self, t):
        return t.transpose(1, 2)


def run_training_loop(*,
                      net, optimizer, train_loader, val_loader,
                      on_epoch, epochs=1, print_interval=5, iterations_per_metric=10):
    criterion = MaskedCrossEntropy()

    for epoch in range(epochs):
        loss_ma = MovingAverage(print_interval)
        pp_ma = MovingAverage(print_interval)

        for i, batch in enumerate(train_loader):
            inputs, labels, mask = batch
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = net(inputs)

            loss = criterion(outputs, labels, mask.mask)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mean_loss = loss_ma(loss.item())
                pp = batch_perplexity(F.softmax(outputs, dim=2), labels, mask.lengths)

                mean_pp = pp_ma(pp)

                if i % print_interval == 0:
                    msg = '\rEpoch {:4}. Iteration {:6}. Loss {:.5}. Perplexity {:.5}'.format(
                        epoch, i, mean_loss, mean_pp
                    )
                    print(msg, end='')

        train_loss = evaluate_loss(net, train_loader, iterations_per_metric)
        val_loss = evaluate_loss(net, val_loader, iterations_per_metric)

        train_pp = evaluate_perplexity(net, train_loader, iterations_per_metric)
        val_pp = evaluate_perplexity(net, val_loader, iterations_per_metric)
        msg = '\rEpoch {:4}. Loss {:8.4f}. Val loss {:8.4f}. Perplexity {:8.4f}. Val perplexity {:8.4f}'
        print(msg.format(epoch, train_loss, val_loss, train_pp, val_pp))
        on_epoch(epoch)


def sample(net, encoder, prompt, steps):
    if prompt.strip():
        tokens = wrapped_tokens(tokenize_prompt(prompt))[:-1]
    else:
        tokens = [wrapped_tokens.start_token]

    classes = encoder.encode_many(tokens)
    num_classes = len(encoder)

    state = None
    output_classes = []
    next_class = classes
    x = one_hot_tensor([next_class], num_classes)
    outputs, state = net(x, state)
    pmf = F.softmax(outputs[0, -1], dim=-1)
    next_class = torch.multinomial(pmf, 1)
    output_classes.append(next_class[0])

    for t in range(steps):
        x = one_hot_tensor([next_class], num_classes)
        outputs, state = net(x, state)
        pmf = F.softmax(outputs[0, -1], dim=-1)
        next_class = torch.multinomial(pmf, 1)

        if encoder.decode(next_class[0]) == wrapped_tokens.end_token:
            break

        output_classes.append(next_class[0])

    return tokens[1:] + encoder.decode_many(output_classes)


def evaluate_loss(net, data_loader, num_iterations=None):
    loss = MaskedCrossEntropy()

    def evaluate_fn(outputs, targets, mask):
        return loss(outputs, targets, mask.mask)

    return evaluate_metric(net, data_loader, evaluate_fn, num_iterations)


def evaluate_perplexity(net, data_loader, num_iterations=None):
    def evaluate_fn(outputs, targets, mask):
        y_hat = F.softmax(outputs, dim=2)
        return batch_perplexity(y_hat, targets, mask.lengths)

    return evaluate_metric(net, data_loader, evaluate_fn, num_iterations)


def evaluate_metric(net, data_loader, evaluate_fn, num_iterations=None):
    num_batches = len(data_loader)
    num_iterations = num_iterations or num_batches

    ma = MovingAverage(num_batches)
    mean_score = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_iterations:
                break

            inputs, targets, mask = batch
            outputs, _ = net(inputs)
            score = evaluate_fn(outputs, targets, mask)
            mean_score = ma(score)

    return mean_score


class ModelStorage:
    model_name = 'model.pt'
    encoding_table_name = 'encoding_table'
    model_params_name = 'model_params'

    @classmethod
    def save(cls, model, model_params, encoder, dir_path):
        os.makedirs(dir_path, exist_ok=True)

        params_json = json.dumps(model_params)
        with open(cls.model_params_path(dir_path), 'w', encoding='utf-8') as f:
            f.write(params_json)

        torch.save(model.state_dict(), cls.model_path(dir_path))
        encoding_table = encoder.serialize()

        with open(cls.encoding_table_path(dir_path), 'w', encoding='utf-8') as f:
            f.write(encoding_table)

    @classmethod
    def model_path(cls, dir_path):
        return os.path.join(dir_path, cls.model_name)

    @classmethod
    def encoding_table_path(cls, dir_path):
        return os.path.join(dir_path, cls.encoding_table_name)

    @classmethod
    def model_params_path(cls, dir_path):
        return os.path.join(dir_path, cls.model_params_name)

    @classmethod
    def load(cls, dir_path):
        if not os.path.isdir(dir_path):
            raise Exception(f'{dir_path} should be a directory')

        model_path = cls.model_path(dir_path)
        params_path = cls.model_params_path(dir_path)
        encoding_table_path = cls.encoding_table_path(dir_path)

        cls._validate_file(model_path)
        cls._validate_file(params_path)
        cls._validate_file(encoding_table_path)

        with open(params_path, 'r', encoding='utf-8') as f:
            params_json = f.read()

        model_params = json.loads(params_json)
        model = Net(**model_params)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with open(encoding_table_path, 'r', encoding='utf-8') as f:
            encoding_table = f.read()

        encoder = Encoder.deserialize(encoding_table)
        return model, encoder

    @classmethod
    def _validate_file(cls, path):
        if not os.path.isfile(path):
            raise Exception(f'File "{path}" does not exist')
