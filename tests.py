import unittest
from preprocessing import tokenize_prompt, paragraph_to_tokens
from preprocessing import Encoder


class TokenizingTests(unittest.TestCase):
    def test(self):
        self.assertEqual([], tokenize_prompt(''))

        self.assertEqual(['Two', 'words'], tokenize_prompt('  Two     words  '))
        self.assertEqual(['Word'], tokenize_prompt('Word'))
        self.assertEqual(['Two', 'words'], tokenize_prompt('Two words'))
        self.assertEqual('More than two words'.split(' '), tokenize_prompt('More than two words'))
        self.assertEqual(['Hello', ',', 'world', '!'], tokenize_prompt('Hello, world!'))

        self.assertEqual([',', 'world', '!'], tokenize_prompt('Hel5lo, world!'))
        self.assertEqual(['Hello', ',', 'world'], tokenize_prompt('Hello, world^'))
        self.assertEqual([',', 'world'], tokenize_prompt('Hel5lo, world^'))

    def test_punctuation(self):
        for punct in '.,?!:;':
            self.assertEqual(['Hello', ',', 'world', punct], tokenize_prompt('Hello, world' + punct))

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

    def test_decode(self):
        encoder = self.encoder
        self.assertEqual('first', encoder.decode(0))
        self.assertEqual('second', encoder.decode(1))
        self.assertEqual('third', encoder.decode(2))
        self.assertEqual(encoder.oov_token, encoder.decode(3))
        self.assertEqual(encoder.oov_token, encoder.decode(-1))
        self.assertEqual(encoder.oov_token, encoder.decode(232))

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


if __name__ == '__main__':
    unittest.main()
