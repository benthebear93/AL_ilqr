import unittest

import numpy as np

from examples.finger_rotate_object import (
    FingerRotationConfig,
    prismatic_from_phi,
    resolved_phi_goal,
    solve_finger_rotation,
)


class TestFingerRotateObject(unittest.TestCase):
    def test_three_link_finger_rotates_disk_clockwise(self):
        config = FingerRotationConfig()
        result = solve_finger_rotation(config)

        self.assertLess(np.linalg.norm(result.q_sol[0] - np.array(config.rolling_initial_q)), 1.0e-8)
        self.assertLess(abs(result.dx_sol[0] - config.prismatic_start), 1.0e-8)
        self.assertLess(abs(result.phi_sol[-1] - resolved_phi_goal(config)), 1.0e-4)
        self.assertLess(abs(result.radial_error[-1]), 1.0e-4)
        self.assertLess(np.linalg.norm(result.q_sol[-1] - result.q_ref[-1]), 1.0e-4)
        self.assertLess(
            abs(result.dx_sol[-1] - prismatic_from_phi(resolved_phi_goal(config), config)),
            1.0e-4,
        )
        self.assertGreater(result.dx_sol[-1], result.dx_sol[0])
