import unittest
import math
from random import shuffle
from preprocessing import tokenize_prompt, paragraph_to_tokens, wrapped_tokens, build_vocab, one_hot_tensor, pad_sequences
from preprocessing import Encoder, ExampleFactory
from utils import MovingAverage
import torch
from metrics import perplexity, batch_perplexity


class TokenizingTests(unittest.TestCase):
    def test(self):
        self.assertEqual([], tokenize_prompt(''))

        self.assertEqual(['two', 'words'], tokenize_prompt('  Two     words  '))
        self.assertEqual(['word'], tokenize_prompt('Word'))
        self.assertEqual(['two', 'words'], tokenize_prompt('Two words'))
        self.assertEqual('more than two words'.split(' '), tokenize_prompt('More than two words'))
        self.assertEqual(['hello', ',', 'world', '!'], tokenize_prompt('Hello, world!'))

        self.assertEqual([',', 'world', '!'], tokenize_prompt('Hel5lo, world!'))
        self.assertEqual(['hello', ',', 'world'], tokenize_prompt('Hello, world^'))
        self.assertEqual([',', 'world'], tokenize_prompt('Hel5lo, world^'))

    def test_punctuation(self):
        for punct in '.,?!:;':
            self.assertEqual(['hello', ',', 'world', punct], tokenize_prompt('Hello, world' + punct))

    def test_paragraph_to_tokens(self):
        sentence1 = ['Hello', ',', 'world', '!']
        sentence2 = ['This', 'is', 'a', 'second', 'sentence', '.']
        paragraph = [sentence1, sentence2]
        self.assertEqual(sentence1 + sentence2, paragraph_to_tokens(paragraph))


class EncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = Encoder('first second third'.split(' '))

    def test_words_must_not_contain_spaces(self):
        self.assertRaises(ValueError, lambda: Encoder(['first', ' second  ', 'third']))

    def test_encode(self):
        encoder = self.encoder
        self.assertEqual(0, encoder.encode('first'))
        self.assertEqual(1, encoder.encode('second'))
        self.assertEqual(2, encoder.encode('third'))
        self.assertEqual(3, encoder.encode('fourth'))
        self.assertEqual(3, encoder.encode('missing'))

    def test_encode_many(self):
        encoder = self.encoder
        encodings = encoder.encode_many(['first', 'second', 'third', 'fourth', 'missing'])
        self.assertEqual([0, 1, 2, 3, 3], encodings)

    def test_decode(self):
        encoder = self.encoder
        self.assertEqual('first', encoder.decode(0))
        self.assertEqual('second', encoder.decode(1))
        self.assertEqual('third', encoder.decode(2))
        self.assertEqual(encoder.oov_token, encoder.decode(3))
        self.assertEqual(encoder.oov_token, encoder.decode(-1))
        self.assertEqual(encoder.oov_token, encoder.decode(232))

    def test_decode_many(self):
        encoder = self.encoder
        decoded_tokens = encoder.decode_many([0, 1, 2, 3, -1, 232])
        self.assertEqual(['first', 'second', 'third'] + [encoder.oov_token] * 3, decoded_tokens)

    def test_round_trip(self):
        encoder = self.encoder
        self.assertEqual('first', encoder.decode(encoder.encode('first')))
        self.assertEqual('second', encoder.decode(encoder.encode('second')))
        self.assertEqual('third', encoder.decode(encoder.encode('third')))
        self.assertEqual(encoder.oov_token, encoder.decode(encoder.encode('missing')))

    def test_build(self):
        tokens = 'first word , then second word , then third word'.split(' ')
        expected_vocab = sorted(set(tokens))
        encoder = Encoder.build(tokens)
        self.assertEqual(expected_vocab, encoder.vocab)

    def test_serialization(self):
        self.assertEqual('first second third', self.encoder.serialize())

        encoder = Encoder.deserialize('first second third')
        self.assertEqual(self.encoder.vocab, encoder.vocab)

    def test_serialization_round_trip(self):
        tokens = tokenize_prompt('Hello, hello, world!')
        original_encoder = Encoder(tokens)
        s = original_encoder.serialize()
        restored_encoder = Encoder.deserialize(s)
        self.assertEqual(restored_encoder.vocab, original_encoder.vocab)

    def test_len_support(self):
        self.assertEqual(4, len(self.encoder))


class ExampleFactoryTests(unittest.TestCase):
    def test_encode(self):
        encoder = Encoder.build(['one', 'two', 'three'])

        tokens = ['one', 'three', 'one', 'oov_word']
        factory = ExampleFactory(tokens, encoder)

        self.assertEqual(tokens, factory.tokens)
        self.assertEqual([encoder.encode(t) for t in tokens], factory.encode())

    def test_make_example(self):
        encoder = Encoder.build(['one', 'two'])
        tokens = ['one', 'two', 'one', 'oov_word']

        factory = ExampleFactory(tokens, encoder)
        x, y = factory.make_example()
        encodings = encoder.encode_many(tokens)
        self.assertEqual(encodings[:-1], x)
        self.assertEqual(encodings[1:], y)


class UtilityTests(unittest.TestCase):
    def test_wrapped_tokens(self):
        tokens = ['Hello', ',', 'world']
        func = wrapped_tokens
        self.assertEqual([func.start_token] + tokens + [func.end_token], func(tokens))

    def test_build_vocab(self):
        tokens = 'apple apple apple apple orange orange orange grape grape tomato'.split(' ')
        shuffle(tokens)

        self.assertEqual([], build_vocab(tokens, max_size=0))
        self.assertEqual([], build_vocab(tokens, max_size=-10))

        self.assertEqual(['apple'], build_vocab(tokens, max_size=1))
        self.assertEqual(['apple', 'orange'], build_vocab(tokens, max_size=2))
        self.assertEqual(['apple', 'grape', 'orange'], build_vocab(tokens, max_size=3))

        self.assertEqual(['apple', 'grape', 'orange', 'tomato'], build_vocab(tokens, max_size=4))
        self.assertEqual(['apple', 'grape', 'orange', 'tomato'], build_vocab(tokens, max_size=5))
        self.assertEqual(['apple', 'grape', 'orange', 'tomato'], build_vocab(tokens, max_size=50))

    def test_one_hot_tensor_error_when_there_are_not_enough_classes(self):
        self.assertRaises(ValueError, lambda: one_hot_tensor([[10, 2, 3]], num_classes=4))

    def test_one_hot_tensor_shape(self):
        classes = [[0, 1, 0, 1], [2, 1, 0, 2], [2, 2, 1, 0]]
        t = one_hot_tensor(classes, num_classes=5)
        self.assertEqual((3, 4, 5), t.shape)

    def test_one_hot_tensor_values(self):
        classes = [[0]]
        t = one_hot_tensor(classes, num_classes=1)

        self.assertTrue(
            torch.allclose(torch.tensor([[1]], dtype=torch.float32), t[0])
        )

        t = one_hot_tensor(classes, num_classes=2)

        self.assertTrue(
            torch.allclose(torch.tensor([[1, 0]], dtype=torch.float32), t[0])
        )

        classes = [[0, 1], [2, 1]]
        actual = one_hot_tensor(classes, num_classes=3)
        expected = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0]],
                [[0, 0, 1], [0, 1, 0]]
            ], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(expected, actual))

    def test_pad_sequences(self):
        seqs = [[0, 1], [2, 3, 4], [5]]
        padded, mask = pad_sequences(seqs, filler=-1)
        self.assertEqual([[0, 1, -1], [2, 3, 4], [5, -1, -1]], padded)

        expected_mask = torch.tensor([[True, True, False], [True, True, True], [True, False, False]])
        self.assertTrue(torch.allclose(expected_mask, mask.mask))
        self.assertEqual([2, 3, 1], mask.lengths)

        seqs = [[1, 2, 3]]
        padded, mask = pad_sequences(seqs, filler=-1)
        self.assertEqual([[1, 2, 3]], padded)
        expected_mask = torch.tensor([[True, True, True]])
        self.assertTrue(torch.allclose(expected_mask, mask.mask))
        self.assertEqual([3], mask.lengths)

    def test_perplexity(self):
        y_hat = torch.tensor([[0, 1, 0]])
        ground_true = torch.tensor([1])
        self.assertAlmostEqual(1, perplexity(y_hat, ground_true), places=7)

        y_hat = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0, 0.5]])
        ground_true = torch.tensor([1, 0])
        self.assertAlmostEqual(1. / math.sqrt(0.1), perplexity(y_hat, ground_true), places=7)

        y_hat = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0, 0.5], [0.25, 0.5, 0.25]])
        ground_true = torch.tensor([1, 0, 2])
        self.assertAlmostEqual(1. / (0.2 * 0.5 * 0.25) ** (1./3), perplexity(y_hat, ground_true), places=7)

    def test_batch_perplexity(self):
        y_hat1 = [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
        y_hat2 = [[0.7, 0.2, 0.1], [0.5, 0, 0.5], [1, 0, 0]]
        y_hat3 = [[0.7, 0.2, 0.1], [0.5, 0, 0.5], [0.25, 0.5, 0.25]]

        targets1 = [1, -1, -1]
        targets2 = [1, 0, -1]
        targets3 = [1, 0, 2]

        y_hat = torch.tensor([y_hat1, y_hat2, y_hat3])
        ground_true = torch.tensor([targets1, targets2, targets3])

        pp1 = perplexity(torch.tensor(y_hat1)[:1], targets1[:1])
        pp2 = perplexity(torch.tensor(y_hat2)[:2], targets2[:2])
        pp3 = perplexity(torch.tensor(y_hat3), targets3)

        expected = (pp1 + pp2 + pp3) / 3.
        actual = batch_perplexity(y_hat, ground_true, [1, 2, 3])
        self.assertAlmostEqual(expected, actual, places=8)


class MovingAverageTest(unittest.TestCase):
    def test_moving_average(self):
        ma = MovingAverage(3)
        self.assertEqual(1, ma(1))
        self.assertEqual(2, ma(3))
        self.assertEqual(3, ma(5))
        self.assertEqual(4, ma(4))


if __name__ == '__main__':
    unittest.main()
