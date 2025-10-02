import unittest
import jax.numpy as jnp
from ilqr.src.costs import Cost, cost, cost_gradient


class TestObjective(unittest.TestCase):
    def test_objective_and_gradients(self):
        T = 3
        num_state = 2
        num_action = 1
        num_parameter = 0

        # define costs
        def ot(x, u, w=None):
            return jnp.dot(x, x) + 0.1 * jnp.dot(u, u)

        def oT(x, u, w=None):
            return 10.0 * jnp.dot(x, x)

        ct = Cost(ot, num_state, num_action, num_parameter=num_parameter)
        cT = Cost(oT, num_state, 0, num_parameter=num_parameter)
        objective = [ct for _ in range(T - 1)] + [cT]

        # test inputs
        x1 = jnp.ones(num_state)
        u1 = jnp.ones(num_action)
        w1 = jnp.zeros(num_parameter)

        X = [x1 for _ in range(T)]
        U = [u1 for _ in range(T - 1)] + [jnp.zeros(0)]
        W = [w1 for _ in range(T)]

        # single-step evaluations
        val_ct = ct.evaluate(x1, u1, w1)
        grad_x_ct = ct.gradient_state(x1, u1, w1)
        grad_u_ct = ct.gradient_action(x1, u1, w1)

        self.assertAlmostEqual(val_ct, ot(x1, u1), places=8)
        self.assertTrue(jnp.allclose(grad_x_ct, 2.0 * x1, atol=1e-8))
        self.assertTrue(jnp.allclose(grad_u_ct, 0.2 * u1, atol=1e-8))

        val_cT = cT.evaluate(x1, jnp.zeros(0), jnp.zeros(0))
        grad_x_cT = cT.gradient_state(x1, jnp.zeros(0), jnp.zeros(0))
        self.assertAlmostEqual(val_cT, oT(x1, jnp.zeros(0)), places=8)
        self.assertTrue(jnp.allclose(grad_x_cT, 20.0 * x1, atol=1e-8))

        # full cost
        J = cost(objective, X, U, W)
        J_expected = sum(ot(X[t], U[t]) for t in range(T - 1)) + oT(X[T - 1], U[T - 1])
        self.assertAlmostEqual(J, J_expected, places=8)

        # gradients
