import os
import unittest
import ngrams
from collections import Counter
import nltk
import numpy as np


class TestSparseArray(unittest.TestCase):
    def test(self):
        sparse = ngrams.SparseArray(6)
        sparse.increment_count(2)
        sparse.increment_count(2)
        sparse.increment_count(5)
        sparse.increment_count(0)
        sparse.increment_count(2)
        sparse.increment_count(5)
        self.assertEqual([1, 0, 3, 0, 0, 2], sparse.to_numpy().tolist())


class CountTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.save_path = 'test_count_table'
        if os.path.isfile(self.save_path):
            os.remove(self.save_path)

    def test_on_uni_grams(self):
        classes = [1, 2, 1, 0, 0, 1]
        ngrams.build_counts_table(classes, num_classes=4, n=1, save_path=self.save_path)
        counts_table = ngrams.CountTable(self.save_path)

        self.assertEqual(1, len(counts_table))

        self.assertEqual([2, 3, 1, 0], counts_table[tuple()].tolist())

    def test_on_bi_grams(self):
        classes = [1, 2, 1, 2, 0, 1, 2]
        ngrams.build_counts_table(classes, num_classes=3, n=2, save_path=self.save_path)
        counts_table = ngrams.CountTable(self.save_path)

        self.assertEqual(3, len(counts_table))
        self.assertEqual([0, 1, 0], counts_table[tuple([0])].tolist())
        self.assertEqual([0, 0, 3], counts_table[tuple([1])].tolist())
        self.assertEqual([1, 1, 0], counts_table[tuple([2])].tolist())

    def test_on_n_grams(self):
        classes1 = [0, 0, 2, 1, 2, 3, 2, 1, 1, 3, 3, 3, 4, 0, 1, 0, 4, 0, 3, 2, 2, 0]
        classes2 = [4, 1, 0, 2, 2, 4, 1, 3, 0, 0, 2, 1, 0, 2, 3, 0, 1, 2, 4, 1, 0, 0]
        classes3 = [0, 1, 2, 3, 4]
        classes4 = [4, 3, 2, 1, 0]

        classes = classes4 + classes1 + classes3 * 2 + classes4 * 2 + classes2 + classes3
        ngrams.build_counts_table(classes, num_classes=5, n=3, save_path=self.save_path)
        counts_table = ngrams.CountTable(self.save_path)

        ngram_counts = Counter(nltk.ngrams(classes, n=3))

        count_rows = {}
        for tri_gram, count in ngram_counts.items():
            *bi_gram, token = tri_gram
            bi_gram = tuple(bi_gram)
            if bi_gram not in count_rows:
                count_rows[bi_gram] = np.zeros(5, dtype=np.int32)
            count_rows[bi_gram][token] = count

        self.assertEqual(len(count_rows), len(counts_table))
        for k in count_rows:
            self.assertEqual(count_rows[k].tolist(), counts_table[k].tolist())
