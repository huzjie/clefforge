import unittest

from clefforge.core.tensor import Tensor, zeros, ones
from clefforge.core.ops import softmax, layernorm, relu


class TestTensor(unittest.TestCase):
    def test_add(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[1, 1], [1, 1]])
        self.assertEqual((a + b).data, [[2, 3], [4, 5]])

    def test_matmul(self):
        a = Tensor([[1, 2, 3]])
        b = Tensor([[1], [1], [1]])
        self.assertEqual((a.matmul(b)).data, [[6.0]])

    def test_softmax_sums_to_one(self):
        p = softmax(Tensor([[1.0, 2.0, 3.0]])).data[0]
        self.assertAlmostEqual(sum(p), 1.0, places=5)

    def test_relu(self):
        self.assertEqual(relu(Tensor([[-1, 2]])).data, [[0.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
