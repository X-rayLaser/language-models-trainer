import torch
from torch import nn
from collections import deque
from metrics import batch_perplexity
from torch.nn import functional as F


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


def run_training_loop(net, optimizer, criterion, dataloader, encoder, epochs=1):
    ma_period = 50
    for epoch in range(epochs):
        mean_loss = 0
        mean_pp = 0
        loss_ma = MovingAverage(ma_period)
        pp_ma = MovingAverage(ma_period)

        for i, batch in enumerate(dataloader):
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

                if i % 25 == 0:
                    msg = '\rEpoch {:4}. Iteration {:6}. Loss {:.5}. Perplexity {:.5}'.format(
                        epoch, i, mean_loss, mean_pp
                    )
                    print(msg, end='')
                    s_from = ' '.join(encoder.decode_many(labels[0]))
                    s_to = ' '.join(encoder.decode_many(outputs[0].argmax(dim=1)))

                    print('\n', s_from, '->')
                    print('', s_to)

        msg = '\rEpoch {:4}. Loss {:.5}. Perplexity {:.5}'.format(epoch, mean_loss, mean_pp)
        print(msg)


def sample(net, encoder, prompt, steps):
    from preprocessing import tokenize_prompt, wrapped_tokens, one_hot_tensor
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
    pmf = F.softmax(outputs[0, -1])
    next_class = torch.multinomial(pmf, 1)
    output_classes.append(next_class[0])

    for t in range(steps):
        x = one_hot_tensor([next_class], num_classes)
        outputs, state = net(x, state)
        pmf = F.softmax(outputs[0, -1])
        next_class = torch.multinomial(pmf, 1)
        output_classes.append(next_class[0])

    return tokens + encoder.decode_many(output_classes)
