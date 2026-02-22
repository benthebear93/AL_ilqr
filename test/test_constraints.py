# ilqr/test/test_constraints.py
import unittest
import numpy as np
from ilqr.src.constraints import Constraint, jacobian_const


class TestConstraints(unittest.TestCase):
    def test_constraints_eval_and_jacobian(self):
        T = 5
        num_state = 2
        num_action = 1
        num_parameter = 0

        # dummy trajectories
        x = [np.array([0.5, -0.2]) for _ in range(T)]
        u = [np.array([0.1]) for _ in range(T - 1)] + [np.zeros(0)]
        w = [np.zeros(num_parameter) for _ in range(T)]

        # define constraints
        def ct(x, u):
            return np.concatenate([-np.ones(num_state) - x, x - np.ones(num_state)])

        def cT(x, u):
            return x

        cont = Constraint(ct, num_state, num_action,
                          indices_inequality=list(range(2 * num_state)),
                          num_parameter=num_parameter)
        conT = Constraint(cT, num_state, 0,
                          indices_inequality=list(range(num_state)),
                          num_parameter=num_parameter)

        constraints = [cont for _ in range(T - 1)] + [conT]

        # check evaluate
        ct0 = cont.evaluate(x[0], u[0], w[0])
        cT0 = conT.evaluate(x[-1], u[-1], w[-1])
        self.assertTrue(np.allclose(ct0, ct(x[0], u[0])))
        self.assertTrue(np.allclose(cT0, cT(x[-1], u[-1])))

        # check jacobians
        jx, ju = jacobian_const(
            [np.zeros((c.num_constraint, c.num_state if t < T - 1 else num_state)) for t, c in enumerate(constraints)],
            [np.zeros((c.num_constraint, c.num_action if t < T - 1 else 0)) for t, c in enumerate(constraints)],
            constraints, x, u, w
        )


        # quick sanity check (no explicit FD check yet)
        for t in range(T - 1):
            self.assertEqual(jx[t].shape, (2 * num_state, num_state))
            self.assertEqual(ju[t].shape, (2 * num_state, num_action))
        # terminal step: no action
        self.assertEqual(jx[-1].shape, (num_state, num_state))
        self.assertEqual(ju[-1].shape[1], 0)  # terminal has no action


if __name__ == "__main__":
    unittest.main()
