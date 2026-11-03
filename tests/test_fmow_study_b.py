import unittest

import torch
from torch import nn

from fyp_drift.evaluation.fmow_study_b import anchor_weights, cloud_image


class StudyBTests(unittest.TestCase):
    def test_anchor_controls_keep_the_same_area(self):
        torch.manual_seed(3)
        tokens = torch.randn(3, 10, 4)
        head = nn.Linear(4, 5)
        saliency = anchor_weights(tokens, head, "saliency", 42)
        shuffled = anchor_weights(tokens, head, "shuffled", 42)
        uniform = anchor_weights(tokens, head, "uniform", 42)
        torch.testing.assert_close(saliency.sum(1), torch.ones(3))
        torch.testing.assert_close(shuffled.sum(1), torch.ones(3))
        torch.testing.assert_close(uniform.sum(1), torch.ones(3))
        self.assertTrue(torch.equal((saliency > 0).sum(1), (shuffled > 0).sum(1)))
        self.assertFalse(torch.equal(saliency, shuffled))

    def test_cloud_is_fixed_per_image_and_seed(self):
        image = torch.zeros(3, 224, 224)
        first = cloud_image(image, "image-a", 0.4, 42)
        torch.testing.assert_close(first, cloud_image(image, "image-a", 0.4, 42))
        self.assertFalse(torch.equal(first, cloud_image(image, "image-b", 0.4, 42)))
        self.assertFalse(torch.equal(first, image))


if __name__ == "__main__":
    unittest.main()
