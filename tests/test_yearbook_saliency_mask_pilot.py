import unittest

import torch

from fyp_drift.training.yearbook_saliency_mask_pilot import mask_images


class YearbookMaskTests(unittest.TestCase):
    def test_mask_keeps_only_selected_equal_area_patches(self):
        image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        weights = torch.tensor([[0.5, 0.0, 0.0, 0.5]])
        masked = mask_images(image, weights)
        expected = image.clone()
        expected[:, :, :2, 2:] = 0
        expected[:, :, 2:, :2] = 0
        self.assertTrue(torch.equal(masked, expected))

    def test_rejects_non_square_patch_grid(self):
        with self.assertRaisesRegex(ValueError, "square grid"):
            mask_images(torch.ones(1, 1, 4, 4), torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
