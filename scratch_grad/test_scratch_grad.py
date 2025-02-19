import unittest

import numpy as np
import numpy.testing as npt
import torch
import torch.nn.functional as F

#from variable import Variable
import sys
sys.path.append('C:\\Users\\HP\\Desktop\\courM2_ulaval\\DL\\les TP\\scratch_grad')

from variable import Variable



class ScratchGradTest(unittest.TestCase):
    def test_add_forward(self):
        a = Variable(3)
        b = Variable(4)
        c = a + b

        self.assertEqual(7, c.data)

    def test_add_backward(self):
        a = Variable(3)
        b = Variable(4)
        c = a + b

        c.backward()

        self.assertEqual(1, a.grad)
        self.assertEqual(1, b.grad)

    def test_add_backward_with_initial_grad(self):
        a = Variable(3)
        b = Variable(4)
        c = a + b

        c.backward(_initial_grad=np.array([13]))

        self.assertEqual(13, a.grad)
        self.assertEqual(13, b.grad)

    def test_sub_forward(self):
        a = Variable(10)
        b = Variable(4)
        c = a - b

        self.assertEqual(6., c.data)

    def test_sub_backward(self):
        a = Variable(3)
        b = Variable(4)
        c = a - b

        c.backward()

        self.assertEqual(1, a.grad)
        self.assertEqual(-1, b.grad)

    def test_sub_backward_with_initial_grad(self):
        a = Variable(3)
        b = Variable(4)
        c = a - b

        c.backward(_initial_grad=np.array([13]))

        self.assertEqual(13, a.grad)
        self.assertEqual(-13, b.grad)

    def test_mul_forward(self):
        a = Variable(7)
        b = Variable(3)
        c = a * b

        self.assertEqual(21, c.data)

    def test_mul_backward(self):
        a = Variable(7.)
        b = Variable(3.)
        c = a * b

        c.backward()

        self.assertEqual(3, a.grad)
        self.assertEqual(7, b.grad)

    def test_mul_backward_with_initial_grad(self):
        a = Variable(7.)
        b = Variable(3.)
        c = a * b

        c.backward(_initial_grad=np.array([13]))

        self.assertEqual(39, a.grad)
        self.assertEqual(91, b.grad)

    def test_div_forward(self):
        a = Variable(12)
        b = Variable(3)
        c = a / b

        self.assertEqual(4., c.data)

    def test_div_backward(self):
        a = Variable(12)
        b = Variable(3)
        c = a / b

        c.backward()

        self.assertEqual(1 / 3, a.grad)
        self.assertEqual(-4 / 3, b.grad)

    def test_div_backward_with_initial_grad(self):
        a = Variable(12)
        b = Variable(3)
        c = a / b

        c.backward(_initial_grad=np.array([12]))

        self.assertEqual(4, a.grad)
        self.assertEqual(-16, b.grad)

    def test_pow_2_backward(self):
        a = Variable(3)
        b = a * a

        b.backward()

        self.assertEqual(6, a.grad)

    def test_neg_forward(self):
        a = Variable(3)
        b = -a

        self.assertEqual(-3, b.data)

    def test_neg_backward(self):
        a = Variable(3)
        b = -a

        b.backward()

        self.assertEqual(-1, a.grad)

    def test_matmul_forward(self):
        a = Variable(np.array([[5, 6]]))
        b = Variable(np.array([[1, 2], [3, 4]]))
        c = a @ b

        expected_data = np.array([[23, 34]])
        npt.assert_almost_equal(expected_data, c.data)

    def test_matmul_backward(self):
        a = Variable(np.array([[5, 6]]))
        b = Variable(np.array([[1, 2], [3, 4]]))
        c = a @ b

        c.backward(_initial_grad=np.array([[1, 1]]))

        expected_grad_a = np.array([[3, 7]])
        expected_grad_b = np.array([[5, 5], [6, 6]])
        npt.assert_almost_equal(a.grad, expected_grad_a)
        npt.assert_almost_equal(b.grad, expected_grad_b)

    def test_mean_row_forward(self):
        a = Variable(np.array([[1.0, 2, 3], [4, 5, 6]]))
        b = a.mean_row()

        expected_data = np.mean(a.data, axis=0)
        npt.assert_almost_equal(b.data, expected_data)

    def test_mean_backward(self):
        a = Variable(np.array([[1.0, 2, 3], [4, 5, 6]]))
        b = a.mean_row()
        b.backward()

        a_torch = torch.tensor([[1.0, 2, 3], [4, 5, 6]], requires_grad=True)
        b_torch = torch.mean(a_torch, dim=0)
        b_torch.backward(torch.ones_like(b_torch))

        expected_grad = a_torch.grad
        npt.assert_almost_equal(a.grad, expected_grad)

    def test_relu_forward(self):
        a = Variable(np.array([[-5, 6]]))
        b = a.relu()

        expected_data = np.array([[0, 6]])
        npt.assert_almost_equal(b.data, expected_data)

    def test_relu_backward(self):
        a = Variable(np.array([[-5, 6]]))
        b = a.relu()

        b.backward()

        expected_grad = np.array([[0, 1]])
        npt.assert_almost_equal(a.grad, expected_grad)

    def test_sigmoid_forward(self):
        a = Variable(np.array([[1, 0]]))
        b = a.sigmoid()

        expected_data = np.array([[1 / (1 + np.exp(-1)), 0.5]])
        npt.assert_almost_equal(b.data, expected_data)

    def test_sigmoid_backward(self):
        a = Variable(np.array([[1, 0]]))
        b = a.sigmoid()

        b.backward()

        expected_grad = np.array([[np.exp(-1) / (1 + np.exp(-1)) ** 2, 0.25]])
        npt.assert_almost_equal(a.grad, expected_grad)

    def test_log_forward(self):
        a = Variable(np.array([1, 10]))
        b = a.log()

        expected_data = np.array([0, np.log(10)])
        npt.assert_almost_equal(b.data, expected_data)

    def test_log_backward(self):
        a = Variable(np.array([[1, 10]]))
        b = a.log()

        b.backward()

        expected_grad = np.array([[1, 0.1]])
        npt.assert_almost_equal(a.grad, expected_grad)

    def test_cross_entropy_forward(self):
        scores = Variable([[1, 2, 0]])
        target = Variable([[0]])
        b = scores.cross_entropy(target)

        a_torch = torch.tensor([[1.0, 2, 0]], requires_grad=True)
        t_torch = torch.tensor([0])
        b_torch = F.cross_entropy(a_torch, t_torch)
        b_torch.backward()

        expected_data = b_torch.data
        npt.assert_almost_equal(b.data, expected_data)

    def test_cross_entropy_forward_numerical_stability(self):
        scores = Variable([[1.0e3, 2, 0]])
        target = Variable([[0]])
        b = scores.cross_entropy(target)

        a_torch = torch.tensor([[1.0e3, 2, 0]], requires_grad=True)
        t_torch = torch.tensor([0])
        b_torch = F.cross_entropy(a_torch, t_torch)
        b_torch.backward()

        expected_data = b_torch.data
        npt.assert_almost_equal(b.data, expected_data)

    def test_cross_entropy_backward(self):
        scores = Variable([[1.0, 5, 0]])
        target = Variable([[0]])
        b = scores.cross_entropy(target)
        b.backward()

        a_torch = torch.tensor([[1.0, 5, 0]], requires_grad=True)
        t_torch = torch.tensor([0])
        b_torch = F.cross_entropy(a_torch, t_torch)
        b_torch.backward()

        expected_grad = a_torch.grad
        npt.assert_almost_equal(scores.grad, expected_grad)

    def test_cross_entropy_backward_numerical_stability(self):
        scores = Variable([[1.0e3, 5, 0]])
        target = Variable([[0]])
        b = scores.cross_entropy(target)
        b.backward()

        a_torch = torch.tensor([[1.0e3, 5, 0]], requires_grad=True)
        t_torch = torch.tensor([0])
        b_torch = F.cross_entropy(a_torch, t_torch)
        b_torch.backward()

        expected_grad = a_torch.grad
        npt.assert_almost_equal(scores.grad, expected_grad)

    def test_small_graph_forward(self):
        a = Variable(3)
        b = Variable(4)
        c = a + b
        d = Variable(1)
        e = c - d
        f = Variable(2)
        g = e * f
        h = Variable(3)
        i = g / h

        self.assertEqual(7, c.data)
        self.assertEqual(6, e.data)
        self.assertEqual(12, g.data)
        self.assertEqual(4, i.data)

    def test_small_graph_backward(self):
        a = Variable(3)
        b = Variable(4)
        c = a + b
        d = Variable(1)
        e = c - d
        f = Variable(2)
        g = e * f
        h = Variable(3)
        i = g / h

        i.backward()

        self.assertEqual(1, i.grad)
        self.assertEqual(-12 / 9, h.grad)
        self.assertEqual(1 / 3, g.grad)
        self.assertEqual(2, f.grad)
        self.assertEqual(2 / 3, e.grad)
        self.assertEqual(-2 / 3, d.grad)
        self.assertEqual(2 / 3, c.grad)
        self.assertEqual(2 / 3, b.grad)
        self.assertEqual(2 / 3, a.grad)

    def test_torch_compatible(self):
        def simple_equation(a, b):
            return (((a + b) + (a * b)) / a - a) @ b @ a

        a = torch.tensor([[3.0, 3.0], [3.0, 3.0]], requires_grad=True)
        b = (torch.eye(2) * 10.0).requires_grad_(True)
        c = simple_equation(a, b)
        c.backward(torch.ones_like(c))

        sg_a = Variable(np.array([[3.0, 3.0], [3.0, 3.0]]))
        sg_b = Variable(np.eye(2) * 10.0)
        sg_c = simple_equation(sg_a, sg_b)
        sg_c.backward(np.ones_like(sg_c.data))

        npt.assert_almost_equal(sg_c.data, c.detach().numpy())
        npt.assert_almost_equal(sg_a.grad, a.grad.detach().numpy(), decimal=5)
        npt.assert_almost_equal(sg_b.grad, b.grad.detach().numpy(), decimal=5)

    def test_net_forward(self):
        w_1_data = np.random.randn(2, 25).astype(np.float32)
        b_1_data = np.random.randn(1, 25).astype(np.float32)
        w_2_data = np.random.randn(25, 2).astype(np.float32)
        b_2_data = np.random.randn(1, 2).astype(np.float32)

        # Torch
        w_1 = torch.tensor(w_1_data, requires_grad=True)
        b_1 = torch.tensor(b_1_data, requires_grad=True)
        w_2 = torch.tensor(w_2_data, requires_grad=True)
        b_2 = torch.tensor(b_2_data, requires_grad=True)

        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([1])

        z_1 = F.relu(x @ w_1 + b_1)
        z_2 = z_1 @ w_2 + b_2
        softmax_out = F.softmax(z_2, dim=1)
        loss = -torch.log(softmax_out[:, y])

        # scratch_grad
        sg_w_1 = Variable(w_1_data, name='w1')
        sg_b_1 = Variable(b_1_data, name='b1')
        sg_w_2 = Variable(w_2_data, name='w2')
        sg_b_2 = Variable(b_2_data, name='b2')

        sg_x = Variable(np.array([[1, 0]]))
        sg_y = Variable(np.array([1]))

        sg_z_1 = (sg_x @ sg_w_1 + sg_b_1).relu()
        sg_z_2 = (sg_z_1 @ sg_w_2 + sg_b_2)

        sg_loss = sg_z_2.cross_entropy(sg_y)

        # Assert scratch_grad is equivalent to PyTorch
        npt.assert_almost_equal(sg_loss.data, loss.detach().numpy().squeeze(), decimal=5)

    def test_net_backward(self):
        w_1_data = np.random.randn(2, 25).astype(np.float32)
        b_1_data = np.random.randn(1, 25).astype(np.float32)
        w_2_data = np.random.randn(25, 2).astype(np.float32)
        b_2_data = np.random.randn(1, 2).astype(np.float32)

        # Torch
        w_1 = torch.tensor(w_1_data, requires_grad=True)
        b_1 = torch.tensor(b_1_data, requires_grad=True)
        w_2 = torch.tensor(w_2_data, requires_grad=True)
        b_2 = torch.tensor(b_2_data, requires_grad=True)

        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([1])

        z_1 = F.relu(x @ w_1 + b_1)
        z_2 = z_1 @ w_2 + b_2
        softmax_out = F.softmax(z_2, dim=1)
        loss = -torch.log(softmax_out[:, y])
        loss.backward()

        # scratch_grad
        sg_w_1 = Variable(w_1_data, name='w1')
        sg_b_1 = Variable(b_1_data, name='b1')
        sg_w_2 = Variable(w_2_data, name='w2')
        sg_b_2 = Variable(b_2_data, name='b2')

        sg_x = Variable(np.array([[1, 0]]))
        sg_y = Variable(np.array([1]))

        sg_z_1 = (sg_x @ sg_w_1 + sg_b_1).relu()
        sg_z_2 = (sg_z_1 @ sg_w_2 + sg_b_2)

        sg_loss = sg_z_2.cross_entropy(sg_y)
        sg_loss.backward()

        # Assert scratch_grad is equivalent to PyTorch
        npt.assert_almost_equal(sg_w_1.data, w_1.detach().numpy(), decimal=5)
        npt.assert_almost_equal(sg_b_1.grad, b_1.grad.detach().numpy(), decimal=5)
        npt.assert_almost_equal(sg_w_2.data, w_2.detach().numpy(), decimal=5)
        npt.assert_almost_equal(sg_b_2.grad, b_2.grad.detach().numpy(), decimal=5)
        npt.assert_almost_equal(sg_loss.data, loss.detach().numpy().squeeze(), decimal=5)


if __name__ == '__main__':
    unittest.main()
