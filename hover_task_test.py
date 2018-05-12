import unittest
from hover_task import HoverTask
import numpy as np

class TestHoverTask(unittest.TestCase):

    def setUp(self):
        self.task = HoverTask()

    def test_hover_target_state(self):
        self.task.sim.pose = np.array([1., 0., 10.,
                                       0., 0., 0.])
        self.task.sim.v = np.array([0., 0., 0.])
        self.task.sim.angular_v = np.array([0., 0., 0.])

        self.assertAlmostEqual(self.task.get_reward(), 0.7)

if __name__ == '__main__':
    unittest.main()
